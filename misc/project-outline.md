# Multi-Instrument Recognition, Mix-Parameter Estimation and Timbre Characterization via Multi-Task Learning (maybe using disentangled representations of instrument sounds + maybe even doing frame-level event detection as well) and Semi-Supervision (as it's already outlined in the Forschungsprämie-Antrag)

Das wäre zumindest mal so grob der Titel für dieses Dokument hier.  
Alex, Christian und ich haben intern mal besprochen was von unserer Seite her interessant wäre, und zwar sind das (wie auch schon im Antrag zur Forschungsprämie beschrieben): 
- Instrument Recognition: Erkennen welche Instrumente in einem Mix vorhanden sind (primär erstmal als globales, "clip-level" label)
- die Charaktersierung von Musik (Instrumenten + Mixes) hinsichtlich deren Klangfarbe: dies vor allem um zu wissen _wie_ ein gegebenes Instrument klingt um anschließend _zB_ nach ähnlich klingenden Tracks zu suchen. 

Zusätzlich wäre Folgendes interessant (wenn auch von geringerer Relevanz):
- Event Detection: instrument recognition / activity detection nicht nur auf globalem clip-level, sondern über den Zeitverlauf, also "event-level"
- Schätzen etwaiger Mix-Parameter (zB relativer Gain der verschiedenen Instrumente zueinander, etc. )

## Literatur
Ich hab mich mal für Literatur bzgl Music Tagging / Classification / Multi-Instrument Recognition und Event Detection + tlw Representation Learning und Pre-Training für diese Anwendungen umgesehen und ein bisschen eingelesen. [Hier](https://www.dropbox.com/sh/4on61kbjtst7v9j/AAAMMrPkKP5mL2Ng95ZgZEeOa?dl=0) ein Dropbox-Link zur Literatur die ich mal da reingeschmissen hab, ist ein bisschen viel - im Unterordner **hot/hothot** sind die momentan relevanten zu finden :)

- 1: Evaluation of CNN-based Automatic Music Tagging Models, Minz Won et al., 2020  
  Gründliche Evaluierung verschiedener Systeme für Music Tagging mit den größeren Datasets dies dafür gibt + Code + Pre-Trained Weights

- 2: Metric Learning vs Classification for Disentangled Music Representation Learning, Jongpil Lee et al., 2020:  
  Fand ich ganz spannend, es geht dabei darum das Classification als Task gut funktioniert um gute Representationen zu lernen (ist jetzt nicht direkt für uns relevant) und darum _disentangled_ Representationen zu lernen (ist für uns zumindest interessant)

- 3: Multi-task Self-supervised Pre-training for Music Classification, Ho-Hsiang Wu et al., 2021:  
  Hier werden klassische Features verwendet als zusätzliche Tasks (für besseres Pre-Training). Detail am Rande: Das Re-Weighting macht definitiv Sinn und kann man sich merken wenn man multi-task learned. 

## Datasets

- Music Tagging (mixes) - (inklusive Tags für Instrumente, also durchaus für Instrument Recognition relevant) - für Details siehe Auflistung in Paper 1:
  - MSD (Million Song Dataset):  
    Audio nicht direkt enthalten, aber kann man sich holen. [Hier](https://github.com/audiolabs/APSRR-2016/tree/master/Bergmann-Raffel) gibts ein script das sich 30s previews "aus dem Web" holen kann. Bzw steht vl auch was im Paper 1. 
  - MagnaTagATune:  
  - Jamendo


* Multi-Instrument Recognition: 
  * [OpenMIC2018](https://github.com/cosmir/openmic-2018)  
  Das ist (denke ich) das einzige brauchbare Dataset das dafür gedacht ist. Wir könnens eventuell als zusätzliche Benchmark verwenden, allerdings werden wir mehr mit den Multi-Tracks arbeiten wollen, weil wir hier die einzelnen Tracks als "Labels" haben und damit deren Event-Activity, Mix-Parameter, und deren Klangfarbe über klassische Features / Deskriptoren. Und wir können uns ziemlich viel Daten selber zusammenmixen wie wir wollen und damit recht viel Diversität erzeugen. 


- Multi-Tracks:
  - [medleyDB](https://medleydb.weebly.com) v1 + v2 
  - [musdb18](https://sigsep.github.io/datasets/musdb.html) (HQ-Version)
  - [SLAKH](http://www.slakh.com/)
  - [mixing secrets / cambridge multi-tracks](https://www.cambridge-mt.com/ms/mtk/)  
    Einige davon sind aber bereits in musdb18 und / oder medleydb enthalten. Außerdem natürlich kein offizielles Dataset in der Community. 
  - unsere hausinternen Multi-Tracks 
  - (musescore, siehe [dieses](https://arxiv.org/abs/1811.01143) paper)

## Approach

Schrittweises Vorgehen wobei zuerst ausgehend von einem Music-Tagging Modell das Multi-Instrument Recognition angegangen wird:

* Step 0: ausgehend vom Code aus Paper 1 deren Ergebnisse reproduzieren. Ist mehr ein optionaler Step aber wahrscheinlich ein guter Einstieg (also nur für das HarmonicCNN und / oder das ShortChunkCNN_Res). 
Außerdem sollte man sich hier wohl die Measures für die Instrument Recognition, die ja teil der Music Tagging Evaluation ist, separat rausrechnen zum Vergleichen mit den folgenden Ergebnissen. 

* Step 0.5: Optional die Performance auf OpenMIC dataset anschaun. Evtl auch mit Vergleich frozen pre-trained weights, finetuned und from scratch. Muss aber nicht sein denke ich..

* Step 1: Eins der pre-trained Modelle verwenden und auf Multi-Track Daten für multi-instrument recognition anwenden (evtl Vergleich: frozen pre-trained weights, finetuned, und from scratch). Welche Daten genau können wir uns noch ausmachen, grundsätzlich wäre medleyDB interessant weil echtes Audiomaterial und relativ viele verschiedene Instrumentklassen. Ist natürlich als alleiniges Dataset relativ klein. Es gibt deswegen auch Leute die verschiedene Datasets kombinieren, also zB medleyDB + musdb18 (siehe "Instrument Activity Detection in Polyphonic Music using Deep Neural Networks, Siddharth Gururani et al., 2018"), also denke ich können wir das auch machen. Also vl die einzelnen Datasets evaluieren, und dann deren Kombination und vergleichen. 

* Step 2: Mix-Parameter schätzen. Im Prinzip das Setup von Step 1 + eben zusätzlich den Gain der einzelnen Sources als Mix-Parameter schätzen (oder RMS oder so). Wie wir den genau definieren so dass es Sinn macht (zB wärs vl klüger das über Loudness zu machen?) können wir uns auch noch ausmachen denke ich, aber die Idee ist wohl klar. Von mir aus auch gerne noch mehr Mix-Parameter, aber mir fällt sonst nicht so viel Sinnvolles ein.  
Spätestens hier (aber wohl auch bei Step 1) sollten wir auf jeden Fall nicht nur die Mixes aus den Datasets nehmen wie sie sind, sondern die Daten "augmentieren", also selbst zusammenmischen. So dass zB mixes entstehen die nur Drums, oder nur Drums + Gesang etc enthalten.

* Step 3: Wir nehmen unser Modell als Backbone und erweitern es mit instrument-spezifischen Layers, fügen also pro Instrument das uns interessiert jeweils Layer hinzu (conv, dense oder attention, mal schaun), so dass wir für jedes Instrument eine eigene Repräsentation erhalten. Nachdem das Modell irgendwie zwischen den Instrumenten unterscheiden muss bzw eine Repräsentation dafür haben muss wenn es (hoffentlich) erfolgreich Instrument Recognition vollziehen kann, möchten wir das nun explizit machen (Nachteil ist das unser Modell nur noch eine fix vorgegebene statische Anzahl an Instrumenten erkennen kann - bzw das ist natürlich vorher auch schon so, aber jetzt haben wir eben für jedes Instrument eigene Layers, vl müssen wir da die Anzahl einschränken (aber die Layers werden nicht grad riesengroß sein müssen, also sollts ok sein)). Jetzt können wir die Recognition von der instrument-spezifischen Repräsentation aus machen - soweit so gut, aber gleichzeitig können wir hier nun die Klangfarbe ins Spiel bringen und uns einen Space aufspannen der die Klangfarbe beschreiben soll. Das können wir - inspiriert von Paper 3 - machen indem wir klassische Features die zB für Brightness, Punchiness etc. stehen (die müssen als Teil der Arbeit natürlich erörtert werden, da aber gibts sehr sehr viel Literatur dazu) hernehmen und als Lernsignal verwenden. Also von diesem Instrument-Space ausgehend fügen wir weitere kleinere Layers hinzu die uns dann schließlich die Werte für eben diese Deskriptoren geben. Als zusätzlichen multi-task Loss haben wir dann zB $L(\hat{y}_{SpectralCentroidDrums}, y_{SpectralCentroidDrums})$ oder auch $L(\hat{y}_{MFCCDrums}, y_{MFCCDrums})$ etc. Und dieser Embedding Space gibt uns dann die Klangfarben Repräsentation für jedes einzelne Instrument im Ausgangsmaterial (idealerweise).  
    ### Disentanglement (siehe Paper 2), die Erste: 
    hier wäre es nett wenn wir diesen instrument-spezifischen Space einigermaßen interpretierbar machen könnten, zb indem wir verschiedene Bereiche in diesem Raum für jeweils verschiedene semantische Klangfarben Deskriptoren stehen. Die einzelnen Layer die dann den jeweiligen Deskriptorwert schätzen könnten also einfach nur einzelne separate Bereiche vom ganzen instrument-spezifischen Embedding Space für die Schätzung berücksichtigen (wir könnten dieses Disentanglement wohl auch über masken direkt mitlernen, denke das könnte aber kompliziert werden, mal schaun). 

    ### Disentanglement die Zweite: 
    Unser Backbone generiert einen Feature-Space, von dem aus dann die instrument-spezifischen Repräsentationen wegbranchen. Optional könnten wir hier ebenfalls die instrument-spezifischen Räume einzelnen Bereichen im vorliegenden Feature Space zuordnen. So könnten wir hierarchische Konzepte widerspiegeln (Idee ist von [hier](https://arxiv.org/abs/2107.07029)), also zB Bereiche in diesem Feature-Space die für vorrangig perkussive Instrumente wichtig sind vs harmonische Instrumente. Dann zB Strings mit den Untergruppen Violine, Cello....  


### Semi-Supervision: 
Das ist denke ich eh schon relativ umfangreich. Allerdings haben wir ja eigentlich vorgehabt das ganze semi-supervised zu machen. Das fällt so wie ich das jetzt skizziert habe allerdings flach. Eine Idee hierzu wäre Folgendes: der supervised Anteil ist das Lernen von instrument-spezifischen Repräsentationen von denen wir die Activity, die Mix-Parameter und eben die Klangfarbe über die Multi-Track Daten lernen. Wir können dann schaun ob diese Repräsentation noch besser wird (hinsichtlich all den unterschiedlichen Aspekten) wenn wir all diese instrument-spezifischen Repräsentationen nehmen, und durch einen weiteren Block / Layer quetschen wovon wir uns dann wiederum eine Repräsentation des Ausgangsmaterial erhoffen dürfen. Was genau wir hier als Lernsignal verwenden können wir uns ausmachen: entweder kommen die auch von den Klangfarben-Deskriptoren des Gesamtsignals, oder wir verwenden die Labels von den Music Tagging Datasets (dann wärs halt nicht so richtig unsupervised an dieser Stelle), oder wir versuchen gar alla Auto-Encoder das Signal (bzw mel-spectrogram) wieder zu rekonstruieren (spätestens hier könnte uns aber der GPU-RAM ausgehen ;) aber wer weiß - nachdem es nur ein zusätzliches Lernsignal sein soll ists uns ja relativ egal ob das rekonstruierte Signal tatsächlich gut rekonstruiert wird, also vl kommen wir hier mit einer kleinen Architektur gut aus). 

## optionales "Stretch-Goal": Event Detection
nicht nur Instrumente erkennen ob sie irgendwo im Signal vorkommen, sondern auch wo im Zeitverlauf sie aktiv sind. Nachdem die pre-trained Modelle für Music Tagging kontinuierlich "poolen", geht die Zeitauflösung flöten. Deshalb müssten wir spätestens hier die Architektur deutlich verändern um den Zeitverlauf abzubilden (darum auch als optionales Ziel, weil des eventuell zu mehr Problemen führen könnte, hyperparameter etc.). Dann könnten wir (zB mit Attention) schauen wo welches Instrument wie stark aktiv ist (vl sogar über RMS Werte), was wir ebenfalls über die Multi-Track Daten schön machen könnten. Außerdem wäre hier die Rekonstruktion des Eingangssignals auf jeden Fall angebracht als unsupervised signal. Und die ganze Klangfarben-Repräsentation + Disentanglement würde ebenfalls dazupassen, aber ist jetzt wohl nicht so wichtig. 
