# coding: utf-8
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import warnings
import random
import torch
from spock import SpockBuilder
from sklearn.preprocessing import LabelBinarizer
from torch.utils.tensorboard import SummaryWriter

from model import ClassPredictor, TimbrePredictor
from config import DataConfig, ModelConfig, TestConfig, TrainConfig, ValidConfig, AugmentationConfig
from data import JamendoDataset, MedleydbDataset, SlakhDataset, MixingSecretsDataset, CombinedDataset
import utils


class Tester:
    def __init__(self, model, model_type, test_dataset, test_config, data_config):
        self.model = model
        self.model_type = model_type
        self.test_dataset = test_dataset
        self.test_loader = self.get_test_loader(test_config)
        self.model_load_path = test_config.model_load_path
        self.samplerate = data_config.samplerate
        self.target_class = data_config.target_class
        self.n_mfccs = data_config.n_mfccs
        self.binarization_threshold = 0.35 #used to convert probabilities from classification output to 1 or 0

        # cuda
        num_of_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_of_gpus}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using Device: {self.device}")

        self.build_model()
        
        if data_config.dataset == 'jamendo':
            self.classes = utils.load_classes(type='jamendo_classes')
            self.mlb = LabelBinarizer().fit(self.classes)
        elif data_config.dataset in ['medleydb', 'slakh', 'mixing-secrets', 'combined']:
            self.instr_families = utils.load_classes(type='instr_families')
            self.instr_classes = utils.load_classes(type='instr_classes')
            self.mlb = LabelBinarizer().fit(self.instr_families + self.instr_classes)

        # Tensorboard
        self.writer = SummaryWriter()
        self.use_tb = test_config.use_tensorboard
        self.tb_rng = np.random.default_rng(test_config.rand_seed)
        self.p_tb = test_config.p_tensorboard


    def get_test_loader(self, test_config):
        def set_test_worker_seed(worker_id):
            """gets called for every worker"""
            worker_seed = test_config.rand_seed +  worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                batch_size=1,
                                                shuffle=False,
                                                drop_last=True,
                                                num_workers=test_config.num_workers,
                                                worker_init_fn=set_test_worker_seed)

        return test_loader


    def build_model(self):
        self.load(self.model_load_path)
        self.model.to(self.device)


    def load(self, filename):
        state_dict = torch.load(filename)
        if 'backbone.spec.mel_scale.fb' in state_dict.keys():
            self.model.backbone.spec.mel_scale.fb = state_dict['backbone.spec.mel_scale.fb']

        self.model.load_state_dict(state_dict)


    def test(self):
        self.model = self.model.eval()
        mean_total_loss, mean_individual_losses = self.get_test_loss()
        print('total_loss: %.4f' % mean_total_loss)
        if self.model_type == 'timbre-predictor':
            print('loudness_loss: %.2f' % round(mean_individual_losses[0], 2))
            print('median_mfcc_loss: %.2f' % round(mean_individual_losses[1], 2))
            print('iqr_mfcc_loss: %.2f' % round(mean_individual_losses[2], 2))
            print('median_spc_loss: %.1f' % round(mean_individual_losses[3], 1))
            print('iqr_spc_loss: %.1f' % round(mean_individual_losses[4], 1))
            print('median_zcr_loss: %.4f' % mean_individual_losses[5])
            print('iqr_zcr_loss: %.4f' % mean_individual_losses[6])
            print('median_spf_loss: %.4f' % mean_individual_losses[7])
            print('iqr_spf_loss: %.4f' % mean_individual_losses[8])


    def scale_predictions(self, out_standardized):
        """scale predictions using pre-computed mean and std
        Note: targets are standardized before training"""
        out_standardized = utils.squeeze_if_tensor_is_3dimensional(out_standardized, dim=0)
        mean = np.loadtxt(f"stats/mean/{self.target_class}.txt")
        std = np.loadtxt(f"stats/std/{self.target_class}.txt")
        mean_tensor = torch.from_numpy(np.vstack([mean] * out_standardized.size(dim=0)))
        std_tensor = torch.from_numpy(np.vstack([std] * out_standardized.size(dim=0)))
        mean_tensor = mean_tensor.to(self.device, dtype=torch.float32)
        std_tensor = std_tensor.to(self.device, dtype=torch.float32)
        out_standardized = out_standardized.to(self.device, dtype=torch.float32)
        out = out_standardized * std_tensor + mean_tensor

        return out


    def get_batch_loss(self, y_tags, y_features, out):
        """compute loss for one batch!"""
        individual_losses = []
        if self.model_type == 'class-predictor':
            total_loss = torch.nn.BCELoss()(out, y_tags)

        elif self.model_type == 'timbre-predictor':
            #apply different weights for each feature
            #weights = np.array([1 for _ in range(1 + self.n_mfccs * 2)] + [0.1, 0.1, 100, 100, 1000, 1000])
            # weights = np.loadtxt(f"weights/{self.target_class}.txt")
            # weight_tensor = torch.from_numpy(np.vstack([weights] * out.size(dim=0)))
            # weight_tensor = weight_tensor.to(self.device, dtype=torch.float32)
            target = utils.squeeze_if_tensor_is_3dimensional(y_features[self.target_class], dim=0)
            target = target.to(self.device, dtype=torch.float32)
            #target = target * weight_tensor
            #out = out * weight_tensor

            #calculate MSE/MAE
            #loss_fct = torch.nn.MSELoss()
            loss_fct = torch.nn.L1Loss()
            total_loss = loss_fct(out, target)
            loudn_loss = loss_fct(out[:, 0], target[:, 0])
            median_mfcc_loss = loss_fct(out[:, 1:self.n_mfccs+1], target[:, 1:self.n_mfccs+1])
            iqr_mfcc_loss = loss_fct(out[:, self.n_mfccs+1:2*self.n_mfccs+1], target[:, self.n_mfccs+1:2*self.n_mfccs+1])
            median_spc_loss = loss_fct(out[:, -6], target[:, -6])
            iqr_spc_loss = loss_fct(out[:, -5], target[:, -5])
            median_zcr_loss = loss_fct(out[:, -4], target[:, -4])
            iqr_zcr_loss = loss_fct(out[:, -3], target[:, -3])
            median_spf_loss = loss_fct(out[:, -2], target[:, -2])
            iqr_spf_loss = loss_fct(out[:, -1], target[:, -1])

            #calculate RMSE
            # total_loss = torch.sqrt(total_loss)
            # loudn_loss = torch.sqrt(loudn_loss)
            # median_mfcc_loss = torch.sqrt(median_mfcc_loss)
            # iqr_mfcc_loss = torch.sqrt(iqr_mfcc_loss)
            # median_spc_loss = torch.sqrt(median_spc_loss)
            # iqr_spc_loss = torch.sqrt(iqr_spc_loss)
            # median_zcr_loss = torch.sqrt(median_zcr_loss)
            # iqr_zcr_loss = torch.sqrt(iqr_zcr_loss)
            # median_spf_loss = torch.sqrt(median_spf_loss)
            # iqr_spf_loss = torch.sqrt(iqr_spf_loss)

            individual_losses.append(loudn_loss.item())
            individual_losses.append(median_mfcc_loss.item())
            individual_losses.append(iqr_mfcc_loss.item())
            individual_losses.append(median_spc_loss.item())
            individual_losses.append(iqr_spc_loss.item())
            individual_losses.append(median_zcr_loss.item())
            individual_losses.append(iqr_zcr_loss.item())
            individual_losses.append(median_spf_loss.item())
            individual_losses.append(iqr_spf_loss.item())

        return total_loss, individual_losses


    def get_test_loss(self):
        out_array = []
        gt_array = []
        total_loss_array = []
        individual_losses_array = np.zeros(1)
        tb_count = 0
        with torch.no_grad():
            for x, y_tags, y_features in tqdm.tqdm(self.test_loader):
                x = utils.squeeze_if_tensor_is_3dimensional(x, dim=0)
                x = x.to(self.device, dtype=torch.float32)
                y_tags = y_tags.squeeze(0).to(self.device, dtype=torch.float32)
                out_standardized = self.model(x)

                out = self.scale_predictions(out_standardized)

                total_loss, individual_losses = self.get_batch_loss(y_tags, y_features, out)
                total_loss_array.append(total_loss.item())
                if individual_losses_array.size == 1:
                    individual_losses_array = np.array(individual_losses)
                else:
                    individual_losses_array = np.vstack((individual_losses_array, individual_losses))

                # log to tensorboard
                if self.use_tb:
                    tb_count = self.tensorboard_log(tb_count, x, out, y_features, individual_losses)

                # average over all chunks in the batch
                if self.model_type == 'class-predictor':
                    out = out.detach().cpu().numpy()
                    out_average = out.mean(axis=0)
                    out_array.append(out_average)
                    ground_truth = y_tags.detach().cpu().numpy()[0,:]
                    gt_array.append(ground_truth)

        if self.model_type == 'class-predictor':
            out_array, gt_array = np.array(out_array), np.array(gt_array)
            out_array_binarized = np.where(out_array < self.binarization_threshold, 0, 1)
            accuracy = utils.get_accuracy(out_array_binarized, gt_array)
            precision = utils.get_precision(out_array_binarized, gt_array)
            recall = utils.get_recall(out_array_binarized, gt_array)
            f1_score = utils.get_f1_score(out_array_binarized, gt_array)
            print('accuracy: %.4f' % np.mean(accuracy))
            print('precision: %.4f' % np.mean(precision))
            print('recall: %.4f' % np.mean(recall))
            print('f1-score: %.4f' % np.mean(f1_score))
            self.plot_metrics(accuracy, precision, recall, f1_score, classes=self.mlb.classes_)
            if utils.check_if_all_classes_exist(gt_array, classes=self.mlb.classes_):
                roc_auc, pr_auc = utils.get_auc(out_array, gt_array)
                print('roc_auc: %.4f' % roc_auc)
                print('pr_auc: %.4f' % pr_auc)
        
        return np.mean(total_loss_array), np.mean(individual_losses_array, axis=0)


    def plot_metrics(self, accuracy, precision, recall, f1_score, classes):
        """plot accuracy, precision, recall and f1-score for each class"""
        x_pos = np.arange(len(classes))
        labels = classes.astype(str).tolist()

        fig, ax = plt.subplots(2, 2, figsize=(10,7))

        # accuracy
        ax[0, 0].bar(x_pos, accuracy)
        ax[0, 0].set_xticks(x_pos)
        ax[0, 0].set_xticklabels(labels, rotation=90)
        ax[0, 0].set(ylabel="Accuracy")

        # precision
        ax[0, 1].bar(x_pos, precision)
        ax[0, 1].set_xticks(x_pos)
        ax[0, 1].set_xticklabels(labels, rotation=90)
        ax[0, 1].set(ylabel="Precision")

        # recall
        ax[1, 0].bar(x_pos, recall)
        ax[1, 0].set_xticks(x_pos)
        ax[1, 0].set_xticklabels(labels, rotation=90)
        ax[1, 0].set(ylabel="Recall")

        # f1-score
        ax[1, 1].bar(x_pos, f1_score)
        ax[1, 1].set_xticks(x_pos)
        ax[1, 1].set_xticklabels(labels, rotation=90)
        ax[1, 1].set(ylabel="F1-Score")

        plt.tight_layout()
        plt.savefig("../../figs/performance.png")


    def tensorboard_log(self, tb_count, x, out, y_features, individual_losses):
        log_to_tb = self.tb_rng.choice([False, True], p=[1 - self.p_tb, self.p_tb])
        if log_to_tb:
            tb_count += 1
            rnd_idx = self.tb_rng.choice(x.size(dim=0))
            self.writer.add_audio('audio_input', x[rnd_idx,:], tb_count, self.samplerate) #NOTE: only adds a single chunk!

            if self.model_type == 'class-predictor':
                fig_1 = self.create_barplot(out[rnd_idx,:]) #TODO: plot mean over all chunks in a batch instead?
                self.writer.add_figure('prediction', fig_1, tb_count)

            elif self.model_type == 'timbre-predictor':
                target_features = utils.squeeze_if_tensor_is_3dimensional(y_features[self.target_class], dim=0)

                # average over all chunks
                estim_features = torch.mean(out, dim=0).tolist()
                target_features = torch.mean(target_features, dim=0).tolist()

                fig_2 = self.plot_predicted_vs_target_mfcc(estim_features, target_features, individual_losses)
                fig_3 = self.plot_predicted_vs_target_features(estim_features, target_features, individual_losses)
                self.writer.add_figure('mfcc', fig_2, tb_count)
                self.writer.add_figure('other_features', fig_3, tb_count)

        return tb_count


    def create_barplot(self, estim_prob):
        result = dict(zip(self.mlb.classes_, estim_prob.tolist()))
        result_list = list(sorted(result.items(), key=lambda x: x[1]))

        fig = plt.figure(figsize=[5, 10])
        plt.barh(np.arange(len(result_list)), [r[1] for r in result_list], align="center")
        plt.yticks(np.arange(len(result_list)), [r[0] for r in result_list])
        plt.tight_layout()

        return fig

    
    def plot_predicted_vs_target_mfcc(self, estim_features, target_features, individual_losses):
        median_mfcc_loss = round(individual_losses[1], 4)
        iqr_mfcc_loss = round(individual_losses[2], 4)

        fig, ax = plt.subplots(2)
        ax[0].plot(estim_features[1:self.n_mfccs+1], 'b', label='estimated')
        ax[0].plot(target_features[1:self.n_mfccs+1], 'r', label='target')
        ax[0].legend()
        ax[0].set_title(f'Median of MFCCs (MAE = {median_mfcc_loss})')
        ax[1].plot(estim_features[self.n_mfccs+1:2*self.n_mfccs+1], 'b', label='estimated')
        ax[1].plot(target_features[self.n_mfccs+1:2*self.n_mfccs+1], 'r', label='target')
        ax[1].legend()
        ax[1].set_title(f'IQR of MFCCs (MAE = {iqr_mfcc_loss})')
        fig.subplots_adjust(hspace=0.5)

        return fig


    def plot_predicted_vs_target_features(self, estim_features, target_features, individual_losses):
        """plot features other than mfcc in a table"""
        #loudness
        estim_loudn = round(estim_features[0], 2)
        target_loudn = round(target_features[0], 2)
        loudn_loss = round(individual_losses[0], 2)

        #spectral centroid
        estim_mean_spc = round(estim_features[-6], 1)
        target_mean_spc = round(target_features[-6], 1)
        estim_iqr_spc = round(estim_features[-5], 1)
        target_iqr_spc = round(target_features[-5], 1)
        mean_spc_loss = round(individual_losses[3], 1)
        iqr_spc_loss = round(individual_losses[4], 1)

        #zero crossing rate
        estim_mean_zcr = round(estim_features[-4], 4)
        target_mean_zcr = round(target_features[-4], 4)
        estim_iqr_zcr = round(estim_features[-3], 4)
        target_iqr_zcr = round(target_features[-3], 4)
        mean_zcr_loss = round(individual_losses[5], 4)
        iqr_zcr_loss = round(individual_losses[6], 4)

        #spectral flatness
        estim_mean_spf = round(estim_features[-2], 5)
        target_mean_spf = round(target_features[-2], 5)
        estim_iqr_spf = round(estim_features[-1], 5)
        target_iqr_spf = round(target_features[-1], 5)
        mean_spf_loss = round(individual_losses[7], 5)
        iqr_spf_loss = round(individual_losses[8], 5)

        data = [[target_loudn, estim_loudn, loudn_loss],
                [target_mean_spc, estim_mean_spc, mean_spc_loss],
                [target_iqr_spc, estim_iqr_spc, iqr_spc_loss],
                [target_mean_zcr, estim_mean_zcr, mean_zcr_loss],
                [target_iqr_zcr, estim_iqr_zcr, iqr_zcr_loss],
                [target_mean_spf, estim_mean_spf, mean_spf_loss],
                [target_iqr_spf, estim_iqr_spf, iqr_spf_loss]]

        fig, ax = plt.subplots(1,1)
        row_labels = ["Loudn", "SPC (med)", "SPC (iqr)", "ZCR (med)", "ZCR (iqr)", "SPF (med)", "SPF (iqr)"]
        column_labels = ["Target", "Estimated", "MAE"]
        #ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=data, rowLabels=row_labels, colLabels=column_labels, cellLoc="center", loc="center", colWidths=[0.20, 0.20, 0.20])
        table.set_fontsize(18)
        table.scale(1.8, 1.8)
        plt.subplots_adjust(left=0.45)

        return fig



def get_datasets(data_config, test_config, aug_config, timb_feat):
    test_datasets = []
    if data_config.dataset == 'combined':
        datasets = ['medleydb', 'slakh', 'mixing-secrets']
    elif data_config.dataset in ['jamendo', 'medleydb', 'slakh', 'mixing-secrets']:
        datasets = data_config.dataset
    else:
        raise ValueError("Dataset has to be one of [jamendo, medleydb, slakh, mixing-secrets, combined]!")

    if data_config.dataset == 'jamendo':
        test_datasets.append(JamendoDataset(data_config, 
                                        split='test', 
                                        chunks_per_track=test_config.chunks_per_track))

    if 'medleydb' in datasets:
        test_datasets.append(MedleydbDataset(data_config,
                                        aug_config,
                                        split='test', 
                                        timb_feat=timb_feat,
                                        samples_per_epoch=test_config.samples_per_epoch,
                                        chunks_per_track=test_config.chunks_per_track))
    
    if 'mixing-secrets' in datasets:
        test_datasets.append(MixingSecretsDataset(data_config,
                                        aug_config,
                                        split='test', 
                                        timb_feat=timb_feat,
                                        samples_per_epoch=test_config.samples_per_epoch,
                                        chunks_per_track=test_config.chunks_per_track))

    if 'slakh' in datasets:
        test_datasets.append(SlakhDataset(data_config,
                                        aug_config,
                                        split='test', 
                                        timb_feat=timb_feat,
                                        samples_per_epoch=test_config.samples_per_epoch,
                                        chunks_per_track=test_config.chunks_per_track))

    if data_config.dataset == 'combined':
        test_dataset = CombinedDataset(medleydb_dataset=test_datasets[0],
                                        mixingsecrets_dataset=test_datasets[1],
                                        slakh_dataset=test_datasets[2],
                                        data_config=data_config, 
                                        split='test',
                                        samples_per_epoch=test_config.samples_per_epoch)
    else:
        test_dataset = test_datasets[0]

    return test_dataset


def get_model(model_config, data_config):
    instr_families = utils.load_classes(type='instr_families')
    instr_classes = utils.load_classes(type='instr_classes')
    n_classes = len(instr_families +  instr_classes)

    if model_config.model_type == 'class-predictor':
        if data_config.dataset == 'jamendo':
            jamendo_classes = utils.load_classes(type='jamendo_classes')
            n_classes = len(jamendo_classes)
        model = ClassPredictor(samplerate=data_config.samplerate,
                                n_classes=n_classes,
                                model_config=model_config)

    elif model_config.model_type == 'timbre-predictor':
        size_of_feature_vectors = 2*(data_config.n_mfccs + 3)
        if data_config.predict_loudness:
            size_of_feature_vectors += 1
        model = TimbrePredictor(samplerate=data_config.samplerate,
                            size_of_feature_vectors=size_of_feature_vectors,
                            model_config=model_config)
                            
    return model


def main():
    config = SpockBuilder(DataConfig,
                        ModelConfig,
                        TrainConfig,
                        ValidConfig,
                        TestConfig,
                        AugmentationConfig).generate()

    data_config = config.DataConfig
    test_config = config.TestConfig
    model_config = config.ModelConfig
    aug_config = config.AugmentationConfig

    # set random seeds for reproducibility
    torch.manual_seed(test_config.rand_seed)
    np.random.seed(test_config.rand_seed)
    random.seed(test_config.rand_seed)

    model_type = model_config.model_type

    # check for 'bad' configurations
    if test_config.num_workers < 4:
        warnings.warn(f'test_config.num_workers is {test_config.num_workers}! Consider using a larger value!')
    if model_type == 'class-predictor':
        if data_config.target_class != 'all-classes':
            warnings.warn('When using the class-predictor the target_class should equal "all-classes"!')
        if data_config.p_skip_percussion == 0.0 or data_config.p_skip_plucked_str == 0.0:
            warnings.warn('When using the class-predictor it is recommended to use values greater than 0.0 for p_skip_percussion and p_skip_plucked_str!')
    elif model_type == 'timbre-predictor':
        if data_config.target_class == 'all-classes' or len(data_config.target_class) == 0:
            warnings.warn('When using the timbre-predictor you have to specify a target_class!')

    timb_feat = True
    if model_type == 'class-predictor':
        timb_feat = False

    test_dataset = get_datasets(data_config, test_config, aug_config, timb_feat)
              
    model = get_model(model_config, data_config)

    tester = Tester(model, model_type, test_dataset, test_config, data_config)
    tester.test()


if __name__ == '__main__':
    main()
