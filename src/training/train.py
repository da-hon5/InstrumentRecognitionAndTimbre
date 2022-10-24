"""
main file for training the neural network
"""
import os
import time
import warnings
import numpy as np
import datetime
import random
import tqdm
import torch
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.preprocessing import LabelBinarizer
from torch.utils.tensorboard import SummaryWriter
from spock import SpockBuilder
from collections import defaultdict

from model import ClassPredictor, TimbrePredictor
from config import DataConfig, ModelConfig, TrainConfig, TestConfig, AugmentationConfig, ValidConfig
from data import JamendoDataset, MedleydbDataset, SlakhDataset, MixingSecretsDataset, CombinedDataset
import utils


class Trainer:
    def __init__(self, model, model_type, train_dataset, valid_dataset, train_config, valid_config, data_config):
        # data loader
        self.model = model
        self.model_type = model_type
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.train_loader = self.get_train_loader(train_config)
        self.valid_loader = self.get_valid_loader(train_config, valid_config)
        self.dataset = data_config.dataset
        self.data_path = data_config.data_path
        self.input_length = data_config.model_input_length * data_config.samplerate
        self.target_class = data_config.target_class
        self.n_mfccs = data_config.n_mfccs
        self.features_array = np.zeros(1) # for calculation of each feature's mean
        self.features_mean = np.zeros(1)
        self.features_std = np.zeros(1)

        # training settings
        self.mode = train_config.mode
        self.n_epochs = train_config.n_epochs
        self.backbone_lr = train_config.backbone_lr
        self.head_lr = train_config.head_lr
        self.lr_scheduler_stepsize = train_config.lr_scheduler_stepsize
        self.lr_scheduler_gamma = train_config.lr_scheduler_gamma
        self.weight_decay = train_config.weight_decay
        self.use_optim_schedule = train_config.use_optim_schedule
        if data_config.dataset == 'jamendo' and not self.use_optim_schedule:
            raise ValueError("To reproduce the results from the paper 'use_optim_schedule' should " 
                            "be True when training with jamendo dataset!")
        
        # model path and step size
        self.model_save_path = train_config.model_save_path
        self.model_load_path = train_config.model_load_path
        self.log_step = train_config.log_step
        self.batch_size = train_config.batch_size

        # cuda
        num_of_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_of_gpus}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using Device: {self.device}")

        # Build model
        self.load_state_dict()
        self.get_optimizer()

        # Tensorboard
        self.writer = SummaryWriter()

        if data_config.dataset == 'jamendo':
            classes = utils.load_classes(type='jamendo_classes')
            self.mlb = LabelBinarizer().fit(classes)
        elif data_config.dataset in ['medleydb', 'slakh', 'mixing-secrets', 'combined']:
            instr_families = utils.load_classes(type='instr_families')
            instr_classes = utils.load_classes(type='instr_classes')
            self.mlb = LabelBinarizer().fit(instr_families + instr_classes)


    def get_train_loader(self, train_config):
        def set_train_worker_seed(worker_id):
            """gets called at every epoch for every worker"""
            # seed has to change for every epoch!!!!
            worker_seed = torch.utils.data.get_worker_info().seed % (2**32 - 1)                                
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                    batch_size=train_config.batch_size,
                                    shuffle=True,
                                    drop_last=True,
                                    num_workers=train_config.num_workers,
                                    worker_init_fn=set_train_worker_seed)

        return train_loader

    
    def get_valid_loader(self, train_config, valid_config):
        def set_valid_worker_seed(worker_id):
            """gets called at every epoch for every worker"""
            # seed has to be the same for every epoch!!!!
            worker_seed = train_config.rand_seed +  worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        valid_loader = torch.utils.data.DataLoader(dataset=self.valid_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    drop_last=True,
                                    num_workers=valid_config.num_workers,
                                    worker_init_fn=set_valid_worker_seed)

        return valid_loader


    def get_optimizer(self):
        backbone_params = [param for name, param in self.model.named_parameters() if 'backbone' in name]
        head_params = [param for name, param in self.model.named_parameters() if 'backbone' not in name]
        
        if self.mode == 'from-scratch':
            if self.backbone_lr != self.head_lr:
                raise ValueError("When training from-scratch 'backbone_lr' and 'head_lr' should be equal (1e-4)")

        self.optimizer = torch.optim.Adam([
                                    {"params": backbone_params, "lr": self.backbone_lr},
                                    {"params": head_params, "lr": self.head_lr}],
                                    weight_decay=self.weight_decay)

        if not self.use_optim_schedule:
            self.scheduler = lr_scheduler.StepLR(self.optimizer, 
                                            step_size=self.lr_scheduler_stepsize, 
                                            gamma=self.lr_scheduler_gamma)


    def load_state_dict(self):
        if self.mode != 'from-scratch':
            # load pretrained model
            state_dict = torch.load(self.model_load_path)
            backbone_dict = {k: v for k, v in state_dict.items() if 'backbone' in k}
            if 'backbone.spec.mel_scale.fb' in state_dict.keys():
                self.model.backbone.spec.mel_scale.fb = state_dict['backbone.spec.mel_scale.fb']
            self.model.load_state_dict(backbone_dict, strict=False)

        # freeze backbone layers ???
        if self.mode == 'frozen-backbone':
            for name, param in self.model.named_parameters():
                if 'backbone' in name:
                    param.requires_grad = False

        self.model.to(self.device)


    def load(self, filename):
        state_dict = torch.load(filename)
        if 'backbone.spec.mel_scale.fb' in state_dict.keys():
            self.model.backbone.spec.mel_scale.fb = state_dict['backbone.spec.mel_scale.fb']
        self.model.load_state_dict(state_dict)


    def compute_stats(self, y_features):
        """compute mean and standard deviation and store them in files prior to training!
        note: this step has to be performed for every class"""
        target = utils.squeeze_if_tensor_is_3dimensional(y_features[self.target_class], dim=0).numpy()
        if self.features_array.size == 1:
            self.features_array = target
        else:
            self.features_array = np.vstack((self.features_array, target))
        self.features_mean = np.mean(self.features_array, axis=0)
        self.features_std = np.std(self.features_array, axis=0)
        
        np.savetxt(f"stats/mean/{self.target_class}.txt", self.features_mean)
        np.savetxt(f"stats/std/{self.target_class}.txt", self.features_std)


    def get_batch_loss(self, y_tags, y_features, out):
        """compute loss for one batch!"""
        if self.model_type == 'class-predictor':
            loss = torch.nn.BCELoss()(out, y_tags)

        elif self.model_type == 'timbre-predictor':
            #NOTE: choose between custom and pre-computed weights
            #weights = np.array([1 for _ in range(1 + self.n_mfccs * 2)] + [0.2, 0.2, 100, 100, 2000, 2000]) #custom weights
            # weights = np.array([2] + 
            #                 [0.3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] + 
            #                 [0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] + 
            #                 [0.05, 0.05, 300, 300, 2000, 2000]) #custom weights
            # weights = np.loadtxt(f"weights/{self.target_class}.txt") #pre-computed weights
            # weights_finetuning = np.array([5] + [0.7 for _ in range(self.n_mfccs * 2)] + [60, 60, 2, 2, 2, 2]) * 100
            # weights = weights * weights_finetuning
            # weight_tensor = torch.from_numpy(np.vstack([weights] * out.size(dim=0)))
            # weight_tensor = weight_tensor.to(self.device, dtype=torch.float32)
            target = y_features[self.target_class] if isinstance(y_features, dict) else y_features
            target = utils.squeeze_if_tensor_is_3dimensional(target, dim=0)
            target = target.to(self.device, dtype=torch.float32)
            # target = target * weight_tensor
            # out = out * weight_tensor
            #loss = torch.nn.MSELoss()(out, target)
            loss = torch.nn.L1Loss()(out, target)
            
        return loss


    def standardize_features(self, y_features):
        """standardize target features using pre-computed mean and std"""
        target = utils.squeeze_if_tensor_is_3dimensional(y_features[self.target_class], dim=0)
        mean = np.loadtxt(f"stats/mean/{self.target_class}.txt")
        std = np.loadtxt(f"stats/std/{self.target_class}.txt")
        mean_tensor = torch.from_numpy(np.vstack([mean] * target.size(dim=0)))
        std_tensor = torch.from_numpy(np.vstack([std] * target.size(dim=0)))
        mean_tensor = mean_tensor.to(self.device, dtype=torch.float32)
        std_tensor = std_tensor.to(self.device, dtype=torch.float32)
        target = target.to(self.device, dtype=torch.float32)
        target = (target - mean_tensor) / std_tensor

        return target


    def train(self):
        start_t = time.time()
        current_optimizer = 'adam'
        best_loss = 999999999999
        drop_counter = 0

        for epoch in range(self.n_epochs):
            iter = 0
            drop_counter += 1
            self.model = self.model.train()  # sets the "mode" to train
            batch_losses = []
            for x, y_tags, y_features in self.train_loader:
                iter += 1

                if self.mode == 'compute-stats':
                    self.compute_stats(y_features)
                    continue
                
                if self.mode != 'compute-stats':
                    x = x.to(self.device, dtype=torch.float32)
                    y_tags = y_tags.to(self.device, dtype=torch.float32)
                    out = self.model(x)

                    y_features_standardized = self.standardize_features(y_features)
                    batch_loss = self.get_batch_loss(y_tags, y_features_standardized, out)
                    batch_losses.append(batch_loss.item())
                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    self.optimizer.step()

                    self.print_log(epoch, iter, batch_loss, start_t)

            if self.mode != 'compute-stats':
                mean_train_loss = np.mean(batch_losses)
                print('mean_train_loss: %.4f' % mean_train_loss)
                self.writer.add_scalar('Loss/train_total', mean_train_loss, epoch)

                # validation
                best_loss = self.validation(best_loss, epoch)

                # lr scheduling
                if self.use_optim_schedule:
                    current_optimizer, drop_counter = self.opt_schedule(current_optimizer, drop_counter)
                else:
                    self.scheduler.step()
                    print(f'learning rate(s): {self.scheduler.get_lr()}')

        print("[%s] Train finished. Elapsed: %s"
                % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    datetime.timedelta(seconds=time.time() - start_t)))


    def opt_schedule(self, current_optimizer, drop_counter):
        """ schedule optimizer! after a certain number of epochs -> change from 
        adam to sgd and decrease learning rate """
        # adam to sgd
        if current_optimizer == 'adam' and drop_counter == 80:
            self.load(os.path.join(self.model_save_path, 'best_model.pth'))
            self.optimizer = torch.optim.SGD(self.model.parameters(), 0.001,
                                            momentum=0.9, weight_decay=0.0001,
                                            nesterov=True)
            current_optimizer = 'sgd_1'
            drop_counter = 0
            print('sgd 1e-3')
        # first drop
        if current_optimizer == 'sgd_1' and drop_counter == 20:
            self.load(os.path.join(self.model_save_path, 'best_model.pth'))
            for pg in self.optimizer.param_groups:
                pg['lr'] = 0.0001
            current_optimizer = 'sgd_2'
            drop_counter = 0
            print('sgd 1e-4')
        # second drop
        if current_optimizer == 'sgd_2' and drop_counter == 20:
            self.load(os.path.join(self.model_save_path, 'best_model.pth'))
            for pg in self.optimizer.param_groups:
                pg['lr'] = 0.00001
            current_optimizer = 'sgd_3'
            print('sgd 1e-5')
        return current_optimizer, drop_counter


    def print_log(self, epoch, iter, loss, start_t):
        if (iter) % self.log_step == 0:
            print("[%s] Epoch [%d/%d] Iter [%d/%d] train_loss: %.4f Elapsed: %s" %
                    (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        epoch+1, self.n_epochs, iter, len(self.train_loader), loss.item(),
                        datetime.timedelta(seconds=time.time()-start_t)))


    def validation(self, best_loss, epoch):
        # the best model is chosen based on the validation loss
        self.model = self.model.eval()
        self.valid_dataset.index_count = defaultdict(int)
        mean_loss = self.get_validation_loss(epoch)
        print('valid_loss_total: %.4f' % mean_loss)
        self.writer.add_scalar('Loss/valid_total', mean_loss, epoch)

       #NOTE: changed this condition (original --> score = 1 - loss)
        if mean_loss < best_loss:
            print('best model!')
            best_loss = mean_loss
            torch.save(self.model.state_dict(), os.path.join(self.model_save_path, 'best_model.pth'))
        return best_loss


    def get_validation_loss(self, epoch):
        """compute validation loss(es) for one epoch"""
        out_array = []
        gt_array = []
        batch_losses = []
        with torch.no_grad():
            for x, y_tags, y_features in tqdm.tqdm(self.valid_loader):
                x = utils.squeeze_if_tensor_is_3dimensional(x, dim=0)
                x = x.to(self.device, dtype=torch.float32)
                y_tags = y_tags.squeeze(0).to(self.device, dtype=torch.float32)
                out = self.model(x)

                y_features_standardized = self.standardize_features(y_features)
                batch_loss = self.get_batch_loss(y_tags, y_features_standardized, out)
                batch_losses.append(batch_loss.item())

                # average over all chunks in the batch
                if self.model_type == 'class-predictor':
                    out = out.detach().cpu().numpy()
                    out_average = out.mean(axis=0)
                    out_array.append(out_average)
                    ground_truth = y_tags.detach().cpu().numpy()[0,:]
                    gt_array.append(ground_truth)

        if self.model_type == 'class-predictor':
            out_array, gt_array = np.array(out_array), np.array(gt_array)
            if utils.check_if_all_classes_exist(gt_array, classes=self.mlb.classes_):
                roc_auc, pr_auc = utils.get_auc(out_array, gt_array)
                print('roc_auc: %.4f' % roc_auc)
                print('pr_auc: %.4f' % pr_auc)
                self.writer.add_scalar('AUC/ROC', roc_auc, epoch)
                self.writer.add_scalar('AUC/PR', pr_auc, epoch)
        
        return np.mean(batch_losses)


def get_datasets(data_config, train_config, valid_config, aug_config, timb_feat):
    train_datasets = []
    valid_datasets = []
    if data_config.dataset == 'combined':
        datasets = ['medleydb', 'slakh', 'mixing-secrets']
    elif data_config.dataset in ['jamendo', 'medleydb', 'slakh', 'mixing-secrets']:
        datasets = data_config.dataset
    else:
        raise ValueError("Dataset has to be one of [jamendo, medleydb, slakh, mixing-secrets, combined]!")

    if data_config.dataset == 'jamendo':
        train_datasets.append(JamendoDataset(data_config, 
                                        split='train'))
        valid_datasets.append(JamendoDataset(data_config, 
                                        split='valid', 
                                        chunks_per_track=valid_config.chunks_per_track))

    if 'medleydb' in datasets:
        train_datasets.append(MedleydbDataset(data_config,
                                        aug_config,
                                        split='train', 
                                        timb_feat=timb_feat,
                                        samples_per_epoch=train_config.samples_per_epoch))
        valid_datasets.append(MedleydbDataset(data_config,
                                        aug_config,
                                        split='valid', 
                                        timb_feat=timb_feat,
                                        samples_per_epoch=valid_config.samples_per_epoch,
                                        chunks_per_track=valid_config.chunks_per_track))
    
    if 'mixing-secrets' in datasets:
        train_datasets.append(MixingSecretsDataset(data_config,
                                        aug_config,
                                        split='train', 
                                        timb_feat=timb_feat,
                                        samples_per_epoch=train_config.samples_per_epoch))
        valid_datasets.append(MixingSecretsDataset(data_config,
                                        aug_config,
                                        split='valid', 
                                        timb_feat=timb_feat,
                                        samples_per_epoch=valid_config.samples_per_epoch,
                                        chunks_per_track=valid_config.chunks_per_track))

    if 'slakh' in datasets:
        train_datasets.append(SlakhDataset(data_config,
                                        aug_config,
                                        split='train', 
                                        timb_feat=timb_feat,
                                        samples_per_epoch=train_config.samples_per_epoch))
        valid_datasets.append(SlakhDataset(data_config,
                                        aug_config,
                                        split='valid', 
                                        timb_feat=timb_feat,
                                        samples_per_epoch=valid_config.samples_per_epoch,
                                        chunks_per_track=valid_config.chunks_per_track))

    if data_config.dataset == 'combined':
        train_dataset = CombinedDataset(medleydb_dataset=train_datasets[0],
                                        mixingsecrets_dataset=train_datasets[1],
                                        slakh_dataset=train_datasets[2],
                                        data_config=data_config, 
                                        split='train',
                                        samples_per_epoch=train_config.samples_per_epoch)
        valid_dataset = CombinedDataset(medleydb_dataset=valid_datasets[0],
                                        mixingsecrets_dataset=valid_datasets[1],
                                        slakh_dataset=valid_datasets[2],
                                        data_config=data_config, 
                                        split='valid',
                                        samples_per_epoch=valid_config.samples_per_epoch)
    else:
        train_dataset = train_datasets[0]
        valid_dataset = valid_datasets[0]

    return train_dataset, valid_dataset


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
                        AugmentationConfig)
    config = config.save()
    config = config.generate()

    data_config = config.DataConfig
    train_config = config.TrainConfig
    valid_config = config.ValidConfig
    model_config = config.ModelConfig
    aug_config = config.AugmentationConfig

    # set random seeds for reproducibility
    torch.manual_seed(train_config.rand_seed)
    np.random.seed(train_config.rand_seed)
    random.seed(train_config.rand_seed)
    
    # create model save dir
    if not os.path.exists(train_config.model_save_path):
        os.makedirs(train_config.model_save_path)

    model_type = model_config.model_type

    # check for 'bad' configurations
    if train_config.num_workers < 4:
        warnings.warn(f'train_config.num_workers is {train_config.num_workers}! Consider using a larger value!')
    if valid_config.num_workers < 4:
        warnings.warn(f'valid_config.num_workers is {valid_config.num_workers}! Consider using a larger value!')
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
        
    train_dataset, valid_dataset = get_datasets(data_config, train_config, valid_config, aug_config, timb_feat)

    model = get_model(model_config, data_config)

    trainer = Trainer(model, model_type, train_dataset, valid_dataset, train_config, valid_config, data_config)
    trainer.train()


if __name__ == '__main__':
    main()