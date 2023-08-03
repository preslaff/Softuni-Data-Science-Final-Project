<a name="br1"></a> 

Unlearning Spurious Correlations in Chest X-ray

Classiﬁcation

Misgina Tsighe Hagos<sup>1,2[0000</sup>−0002−9318−<sup>9417]</sup>, Kathleen M.

Curran<sup>1,3[0000</sup>−0003−0095−<sup>9337]</sup>, and Brian Mac Namee<sup>1,2[0000</sup>−0003−2518−0274]

<sup>1</sup> Science Foundation Ireland Centre for Research Training in Machine Learning

misgina.hagos@ucdconnect.ie

<sup>2</sup> School of Computer Science, University College Dublin, Ireland

brian.macnamee@ucd.ie

<sup>3</sup> School of Medicine, University College Dublin, Ireland

kathleen.curran@ucd.ie

Abstract. Medical image classiﬁcation models are frequently trained

using training datasets derived from multiple data sources. While lever-

aging multiple data sources is crucial for achieving model generaliza-

tion, it is important to acknowledge that the diverse nature of these

sources inherently introduces unintended confounders and other chal-

lenges that can impact both model accuracy and transparency. A no-

table confounding factor in medical image classiﬁcation, particularly in

musculoskeletal image classiﬁcation, is skeletal maturation-induced bone

growth observed during adolescence. We train a deep learning model us-

ing a Covid-19 chest X-ray dataset and we showcase how this dataset

can lead to spurious correlations due to unintended confounding regions.

eXplanation Based Learning (XBL) is a deep learning approach that

goes beyond interpretability by utilizing model explanations to inter-

actively unlearn spurious correlations. This is achieved by integrating

interactive user feedback, speciﬁcally feature annotations. In our study,

we employed two non-demanding manual feedback mechanisms to imple-

ment an XBL-based approach for eﬀectively eliminating these spurious

correlations. Our results underscore the promising potential of XBL in

constructing robust models even in the presence of confounding factors.

Keywords: Interactive Machine Learning · eXplanation Based Learning

· Medical Image Classiﬁcation · Chest X-ray

1

Introduction

While Computer-Assisted Diagnosis (CAD) holds promise in terms of cost and

time savings, the performance of models trained on datasets with undetected

biases is compromised when applied to new and external datasets. This limitation

hinders the widespread adoption of CAD in clinical practice [\[21,16\].](#br11)[ ](#br11)Therefore,

it is crucial to identify biases within training datasets and mitigate their impact

on trained models to ensure model eﬀectiveness.



<a name="br2"></a> 

2

M. T Hagos et al.

Fig. 1. In the left image, representing a child diagnosed with Viral pneumonia, the

presence of Epiphyses on the humerus heads is evident, highlighted with red ellipses.

Conversely, the right image portrays an adult patient with Covid-19, where the Epi-

physes are replaced by Metaphyses, also highlighted with red ellipses.

For example, when building models for the diﬀerential diagnosis of pathology

on chest X-rays (CXR) it is important to consider skeletal growth or ageing as

a confounding factor. This factor can introduce bias into the dataset and poten-

tially mislead trained models to prioritize age classiﬁcation instead of accurately

distinguishing between speciﬁc pathologies. The eﬀect of skeletal growth on the

appearance of bones necessitates careful consideration to ensure that a model

focuses on the intended classiﬁcation task rather than being inﬂuenced by age-

related features.

An illustrative example of this scenario can be found in a recent study by

Pfeuﬀer et al. [\[12\].](#br10)[ ](#br10)In their research, they utilized the Covid-19 CXR dataset

[\[4\],](#br10)[ ](#br10)which includes a category comprising CXR images of children. This dataset

serves as a pertinent example to demonstrate the potential inﬂuence of age-

related confounders, given the presence of images from pediatric patients. It

comprises CXR images categorized into four groups: Normal, Covid, Lung opac-

ity, and Viral pneumonia. However, a notable bias is introduced into the dataset

due to the speciﬁc inclusion of the Viral pneumonia cases collected exclusively

from children aged one to ﬁve years old [\[9\].](#br10)[ ](#br10)This is illustrated in Figure [1](#br2)[ ](#br2)where

confounding regions introduced due to anatomical diﬀerences between a child

and an adult in CXR images are highlighted. Notably, the presence of Epiphyses

in images from the Viral pneumonia category (which are all from children) is a

confounding factor, as it is not inherently associated with the disease but can

potentially mislead a model into erroneously associating it with the category.

Addressing these anatomical diﬀerences is crucial to mitigate potential bias and

ensure accurate analysis and classiﬁcation in pediatric and adult populations.

Biases like this one pose a challenge to constructing transparent and robust

models capable of avoiding spurious correlations. Spurious correlations refer to

image regions that are mistakenly believed by the model to be associated with

a speciﬁc category, despite lacking a genuine association.

While the exact extent of aﬀected images remains unknown, it is important

to note that the dataset also encompasses other confounding regions, such as



<a name="br3"></a> 

Unlearning Spurious Correlations in Chest X-ray Classiﬁcation

3

texts and timestamps. However, it is worth mentioning that these confounding

regions are uniformly present across all categories, indicating that their impact is

consistent throughout. For the purpose of this study, we speciﬁcally concentrate

on understanding and mitigating the inﬂuence of musculoskeletal age in the

dataset.

eXplanation Based Learning (XBL) represents a branch of Interactive Ma-

chine Learning (IML) that incorporates user feedback in the form of feature

annotation during the training process to mitigate the inﬂuence of confounding

regions [\[17\]](#br11). By integrating user feedback into the training loop, XBL enables

the model to progressively improve its performance and enhance its ability to

diﬀerentiate between relevant and confounding features [\[6\].](#br10)[ ](#br10)In addition to un-

learning spurious correlations, XBL has the potential to enhance users’ trust in

a model [\[5\].](#br10)[ ](#br10)By actively engaging users and incorporating their expertise, XBL

promotes a collaborative learning environment, leading to increased trust in the

model’s outputs. This enhanced trust is crucial for the adoption and acceptance

of models in real-world applications, particularly in domains where decisions

have signiﬁcant consequences, such as medical diagnosis.

XBL approaches typically add regularization to the loss function used when

training a model, enabling it to disregard the impact of confounding regions. A

typical XBL loss can be expressed as:

X

2

(1)

L = L<sub>CE</sub> + L<sub>expl</sub> + λ

θ ,

i

i=0

where L<sub>CE</sub> is categorical cross entropy loss that measures the discrepancy be-

tween the model’s predictions and ground-truth labels; λ is a regularization term;

θ refers to network parameters; and L<sub>expl</sub> is an explanation loss. Explanation

loss can be formulated as:

X<sup>N</sup>

L<sub>expl</sub>

\=

M ⊙ Exp(x ) ,

(2)

i

i

i=0

where N is the number of training instances, x ∈ X; M is a manual annota-

i

tion of confounding regions in the input instance x ; and Exp(x ) is a saliency-

i

i

based model explanation for instance x<sub>i</sub>, for example generated using Gradient

weighted Class Activation Mapping (GradCAM) [\[17\].](#br11)[ ](#br11)GradCAM is a feature

attribution based model explanation that computes the attention of the learner

model on diﬀerent regions of an input image, indicating the regions that sig-

niﬁcantly contribute to the model’s predictions [\[18\].](#br11)[ ](#br11)This attention serves as

a measure of the model’s reliance on these regions when making predictions.

The loss function, L<sub>expl</sub>, is designed to increase as the learner’s attention to

the confounding regions increases. Overall, by leveraging GradCAM-based at-

tention and the associated L<sub>expl</sub> loss, XBL provides a mechanism for reducing

a model’s attention to confounding regions, enhancing the interpretability and

transparency of a model’s predictions.

As is seen in the inner ellipse of Figure [2,](#br4)[ ](#br4)in XBL, the most common mode

of user interaction is image feature annotation. This requires user engagement



<a name="br4"></a> 

4

M. T Hagos et al.

Good explanation

Bad explanation

Ranking

as feedback

Feature

annotation as

feedback

Focus on the annotated

image regions

Fig. 2. The inner ellipse shows the typical mode of feedback collection where users

annotate image features. The outer ellipse shows how our proposed approach requires

only identiﬁcation of one good and one bad explanation.

that is considerably more demanding than the simple instance labeling that most

IML techniques require [\[22\]](#br11)[ ](#br11)and increases the time and cost of feedback collec-

tion. As can be seen in the outer ellipse of Figure [2,](#br4)[ ](#br4)we are interested in lifting

this pressure from users (feedback providers) and simplifying the interaction to

ask for identiﬁcation of two explanations as exemplary explanations and rank-

ing them as good and bad explanations. This makes collecting feedback cheaper

and faster. This kind of user interaction where users are asked for a ranking in-

stead of category labels has also been found to increase inter-rater reliability and

data collection eﬃciency [\[11\].](#br10)[ ](#br10)We incorporate this feedback into model training

through a contrastive triplet loss [\[3\].](#br10)

The main contributions of this paper are:

1\. We propose the ﬁrst type of eXplanation Based Learning (XBL) that can

learn from only two exemplary explanations of two training images;

2\. We present an approach to adopt triplet loss for XBL to incorporate the two

exemplary explanations into an explanation loss;

3\. Our experiments demonstrate that the proposed method achieves improved

explanations and comparable classiﬁcation performance when compared

against a baseline model.

2

Related Work

2\.1 Chest x-ray classiﬁcation

A number of Covid-19 related datasets have been collated and deep learning

based diagnosis solutions have been proposed due to the health emergency caused

by Covid-19 and due to an urgent need for computer-aided diagnosis (CAD)

of the disease [\[8\].](#br10)[ ](#br10)In addition to training deep learning models from scratch,



<a name="br5"></a> 

Unlearning Spurious Correlations in Chest X-ray Classiﬁcation

5

transfer learning, where parameters of a pre-trained model are further trained

to identify Covid-19, have been utilized [\[20\].](#br11)[ ](#br11)Even though the array of datasets

and deep learning models show promise in implementing CAD, care needs to

be taken when the datasets are sourced from multiple imaging centers and/or

the models are only validated on internal datasets. The Covid-19 CXR dataset,

for example, has six sources at the time of writing this paper. This can result

in unintended confounding regions in images in the dataset and subsequently

spurious correlations in trained models [\[16\].](#br11)

2\.2 eXplanation Based Learning

XBL can generally be categorized based on how feedback is used: (1) augmenting

loss functions; and (2) augmenting training datasets.

Augmenting Loss Functions. As shown in Equation [1,](#br3)[ ](#br3)approaches in this cate-

gory add an explanation loss, L<sub>expl</sub>, during model training to encourage focus

on image regions that are considered relevant by user(s), or to ignore confound-

ing regions [\[7\].](#br10)[ ](#br10)Ross et al. [\[14\]](#br10)[ ](#br10)use an L<sub>expl</sub> that penalizes a model with high

input gradient model explanations on the wrong image regions based on user

annotation,

"

\#

2

X<sup>N</sup>

X<sup>K</sup>

∂

L<sub>expl</sub>

\=

M<sub>n</sub> ⊙

log yˆ<sub>nk</sub>

,

(3)

∂x<sub>n</sub>

n

k=1

for a function f(X|θ) = yˆ ∈ R<sup>N</sup>×<sup>K</sup> trained on N images, x , with K categories,

n

where M ∈ [{](#br11)0, 1} is user annotation of confounding image regions. Similarly,

n

Shao et al. [\[19\]](#br11)[ ](#br11)use inﬂuence functions in place of input gradients to correct a

model’s behavior

Augmenting Training Dataset. In this category, a confounder-free dataset is

added to an existing confounded training dataset to train models to avoid learn-

ing spurious correlations. In order to unlearn spurious correlations from a classi-

ﬁer that was trained on the Covid-19 dataset, Pfeuﬀer et al. [\[12\]](#br10)[ ](#br10)collected feature

annotation on 3,000 chest x-ray images and augmented their training dataset.

This approach, however, doesn’t target unlearning or removing spurious corre-

lations, but rather adds a new variety of data. This means models are being

trained on a combination of the existing confounded training dataset and the

their new dataset.

One thing all approaches to XBL described above have in common is the

assumption that users will provide feature annotation for all training instances

to reﬁne or train a model. We believe that this level of user engagement hinders

practical deployment of XBL because of the demanding nature and expense of

feature annotation that is required [\[22\].](#br11)[ ](#br11)It is, therefore, important to build an

XBL method that can reﬁne a trained model using a limited amount of user

interaction and we propose eXemplary eXplanation Based Learning to achieve

this.



<a name="br6"></a> 

6

M. T Hagos et al.

3

eXemplary eXplanation Based Learning

As illustrated in Equation [1,](#br3)[ ](#br3)for typical XBL approaches, user annotation of

image features, or M, is an important prerequisite. We propose eXemplary eX-

planation Based Learning (eXBL) to mitigate the time and resource complexity

caused by the need for M. In eXBL, we simplify the expensive feature anno-

tation requirement and replace it with identiﬁcation of just two exemplary ex-

planations: a Good explanation (C<sub>goodi</sub> ) and a Bad explanation (C<sub>badj</sub> ), of two

diﬀerent instances, x and x . We pick the two exemplary explanations manually

i

j

based on how much attention a model’s explanation output gives to relevant im-

age regions. A good explanation would be one that gives more focus to the lung

and chest area rather than the irrelevant regions such as the Epiphyses, humerus

head, and image backgrounds, while a bad explanation does the opposite.

We choose to use GradCAM model explanations because they have been

found to be more sensitive to training label reshuﬄing and model parameter

randomization than other saliency based explanations [\[1\];](#br10)[ ](#br10)and they provide ac-

curate explanations in medical image classiﬁcations [\[10\].](#br10)

We then compute product of the input instances and the Grad-CAM expla-

nation in order to propagate input image information towards computing the

loss and to avoid a bias that may be caused by only using a model’s GradCAM

explanation,

C<sub>good</sub> := x<sub>i</sub> ⊙ C<sub>good</sub>

C<sub>bad</sub> := x<sub>j</sub> ⊙ C<sub>bad</sub>

(4)

(5)

i

j

We then take inspiration from triplet loss [\[3\]](#br10)[ ](#br10)to incorporate C<sub>good</sub> and C<sub>bad</sub>

into our explanation loss, L<sub>expl</sub>. The main purpose of L<sub>expl</sub> is to penalize a

trainer according to similarity of model explanations of instance x to C<sub>good</sub> and

its diﬀerence from C<sub>bad</sub>. We use Euclidean distance as a loss to compute the

measure of dissimilarity, d (loss decreases as similarity to C<sub>good</sub> is high and to

C<sub>bad</sub> is low).

d<sub>xg</sub> := d(x ⊙ GradCAM(x), C<sub>good</sub>

d<sub>xb</sub> := d(x ⊙ GradCAM(x), C<sub>bad</sub>

)

(6)

(7)

)

We train the model f to achieve d<sub>xg</sub> ≪ d for all x. We do this by adding a

xb

xb

margin = 1.0 and translating it to: d<sub>xg</sub> < d + margin. We then compute the

explanation loss as:

X<sup>N</sup>

L<sub>expl</sub>

\=

max(d<sub>xig</sub> − d<sub>xib</sub> + margin, 0)

(8)

i

In addition to correctly classifying X, which is achieved through L<sub>CE</sub>, this

L<sub>expl</sub> (Equation [8)](#br6)[ ](#br6)trains f to output GradCAM values that resemble the good

explanations [and](#br6)[ ](#br6)that diﬀer from the bad explanations, thereby reﬁning the

model to focus on the relevant regions and to ignore confounding regions. L<sub>expl</sub>



<a name="br7"></a> 

Unlearning Spurious Correlations in Chest X-ray Classiﬁcation

7

is zero, for a given sample x, unless x ⊙ GradCAM(x) is much more similar to

bad

C

than it is to C

—meaning d > d + margin.

good xg xb

4

Experiments

4\.1 Data Collection and Preparation

To demonstrate eXBL we use the Covid-19 CXR dataset [\[4,13\]](#br10)[ ](#br10)described in

Section [1.](#br1)[ ](#br1)For model training we subsample 800 x-ray images per category to

mitigate class imbalance, totaling 3,200 images. For validation and testing, we

use 1,200 and 800 images respectively. We resize all images to 224 × 224 pixels.

The dataset is also accompanied with feature annotation masks that show the

lungs in each of the x-ray images collected from radiologists [\[13\].](#br10)

4\.2 Model Training

We followed a transfer learning approach using a pre-trained MobileNetV2 model

[\[15\].](#br11)[ ](#br11)We chose to use MobileNetV2 because it achieved better performance at

the CXR images classiﬁcation task at a reduced computational cost after com-

parison among pre-trained models. In order for the training process to aﬀect

the GradCAM explanation outputs, we only freeze and reuse the ﬁrst 50 layers

of MobileNetV2 and retrain the rest of the convolutional layers with a custom

classiﬁer layer that we added (256 nodes with a ReLu activation with a 50%

dropout followed by a Softmax layer with 4 nodes).

We ﬁrst trained the MobileNetV2 to categorize the training set into the four

classes using categorical cross entropy loss. It was trained for 60 epochs[<sup>4</sup>](#br7). We refer

to this model as the Unreﬁned model. We then use the Unreﬁned model to select

the good and bad explanations displayed in Figure [2.](#br4)[ ](#br4)Next, we employ our eXBL

algorithm using the good and bad explanations to teach the Unreﬁned model to

focus on relevant image regions by tuning its explanations to look like the good

explanations and to diﬀer from the bad explanations as much as possible. We use

Euclidean distance to compute dissimilarity in adopting a version of the triplet

loss for XBL. We refer to this model as the eXBL<sub>EUC</sub> model and it was trained

for 100 epochs using the same early stopping, learning rate, and optimizer as

the Unreﬁned model.

For model evaluation, in addition to classiﬁcation performance, we compute

an objective explanation evaluation using Activation Precision [\[2\]](#br10)[ ](#br10)that measures

how many of the pixels predicted as relevant by a model are actually relevant

using existing feature annotation of the lungs in the employed dataset,

P

X<sup>N</sup>

1

(T (GradCAM (x )) ⊙ A

)

τ

θ

n

<sup>x</sup>n

(9)

AP =

P

,

N

(T (GradCAM (x )))

τ

θ

n

n

<sup>4</sup> The model was trained with an early stop monitoring the validation loss at a patience

of ﬁve epochs and a decaying learning rate = 1e-04 using an Adam optimizer.



<a name="br8"></a> 

8

M. T Hagos et al.

where x<sub>n</sub> is a test instance, A<sub>xn</sub> is feature annotation of lungs in the dataset,

GradCAM (x ) holds the GradCAM explanation of x generated from a trained

θ

n

n

model, and T<sub>τ</sub> is a threshold function that ﬁnds the (100-τ) percentile value and

sets elements of the explanation, GradCAM (x ), below this value to zero and

θ

n

the remaining elements to one. In our experiments, we use τ = 5%.

A

B

C

D

E

Fig. 3. Sample outputs of Viral Pneumonia category. (A) Input images; (B) GradCAM

outputs for Unreﬁned model and (C) their overlay over input images; (D) GradCAM

outputs for eXBL<sub>EUC</sub> and (E) their overlay over input images.



<a name="br9"></a> 

Unlearning Spurious Correlations in Chest X-ray Classiﬁcation

9

Table 1. Classiﬁcation and explanation performance.

Accuracy

Activation Precision

Models

Validation Test Validation

Test

0\.32

0\.35

Unreﬁned

eXBL<sub>EUC</sub>

0\.94

0\.89

0\.95

0\.90

0\.32

0\.34

5

Results

Table [1](#br9)[ ](#br9)shows classiﬁcation and explanation performance of the Unreﬁned and

eXBL<sub>EUC</sub> models. Sample test images, GradCAM outputs, and overlaid Grad-

CAM visualizations of x-ray images with Viral pneumonia category are displayed

in Figure [3.](#br8)[ ](#br8)From the sample GradCAM outputs and Table [1,](#br9)[ ](#br9)we observe that

the eXBL<sub>EUC</sub> model was able to produce more accurate explanations that avoid

focusing on irrelevant image regions such as the Epiphyses and background re-

gions. This is demonstrated by how GradCAM explanations of the eXBL<sub>EUC</sub>

model tend to focus on the central image regions of the input images focusing

around the chest that is relevant for the classiﬁcation task, while the GradCAM

explanations generated using the Unreﬁned model give too much attention to ar-

eas around the shoulder joint (humerus head) and appear angular shaped giving

attention to areas that are not related with the disease categories.

6

Conclusion

In this work, we have presented an approach to debug a spurious correlation

learned by a model and to remove it with just two exemplary explanations in

eXBL<sub>EUC</sub>. We present a way to adopt the triplet loss for unlearning spurious cor-

relations. Our approach can tune a model’s attention to focus on relevant image

regions, thereby improving the saliency-based model explanations. We believe

it could be easily adopted to other medical or non-medical datasets because it

only needs two non-demanding exemplary explanations as user feedback.

Even though the eXBL<sub>EUC</sub> model achieved improved explanation perfor-

mances when compared to the Unreﬁned model, we observed that there is a

classiﬁcation performance loss when retraining the Unreﬁned model with eXBL

to produce good explanations. This could mean that the initial model was ex-

ploiting the confounding regions for better classiﬁcation performance. It could

also mean that our selection of good and bad explanations may not have been

optimal and that the two exemplary explanations may be degrading model per-

formance.

Since our main aim in this study was to demonstrate eﬀectiveness of

eXBL<sub>EUC</sub> based on just two ranked feedback, the generated explanations were

evaluated using masks of lung because it is the only body part with pixel-level

annotation in the employed dataset. However, in addition to the lung, the disease

categories might be associated with other areas of the body such as the throat

and torso. For this reason, and to ensure transparency in practical deployment

of such systems in clinical practice, future work should involve expert end users

for evaluation of the classiﬁcation and model explanations.



<a name="br10"></a> 

10

M. T Hagos et al.

Acknowledgements

This publication has emanated from research conducted with the ﬁnancial sup-

port of Science Foundation Ireland under Grant number 18/CRT/6183. For the

purpose of Open Access, the author has applied a CC BY public copyright licence

to any Author Accepted Manuscript version arising from this submission.

References

1\. Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., Kim, B.: Sanity

checks for saliency maps. arXiv preprint arXiv:1810.03292 (2018)

2\. Barnett, A.J., Schwartz, F.R., Tao, C., Chen, C., Ren, Y., Lo, J.Y., Rudin, C.: A

case-based interpretable deep learning model for classiﬁcation of mass lesions in

digital mammography. Nature Machine Intelligence 3(12), 1061–1070 (2021)

3\. Chechik, G., Sharma, V., Shalit, U., Bengio, S.: Large scale online learning of image

similarity through ranking. Journal of Machine Learning Research 11(3) (2010)

4\. Chowdhury, M.E., Rahman, T., Khandakar, A., Mazhar, R., Kadir, M.A., Mahbub,

Z.B., Islam, K.R., Khan, M.S., Iqbal, A., Al Emadi, N., et al.: Can ai help in

screening viral and covid-19 pneumonia? IEEE Access 8, 132665–132676 (2020)

5\. Dietvorst, B.J., Simmons, J.P., Massey, C.: Overcoming algorithm aversion: People

will use imperfect algorithms if they can (even slightly) modify them. Management

Science 64(3), 1155–1170 (2018)

6\. Hagos, M.T., Curran, K.M., Mac Namee, B.: Identifying spurious correla-

tions and correcting them with an explanation-based learning. arXiv preprint

arXiv:2211.08285 (2022)

7\. Hagos, M.T., Curran, K.M., Mac Namee, B.: Impact of feedback type on explana-

tory interactive learning. In: International Symposium on Methodologies for Intel-

ligent Systems. pp. 127–137. Springer (2022)

8\. Islam, M.M., Karray, F., Alhajj, R., Zeng, J.: A review on deep learning techniques

for the diagnosis of novel coronavirus (covid-19). Ieee Access 9, 30551–30572 (2021)

9\. Kermany, D.S., Goldbaum, M., Cai, W., Valentim, C.C., Liang, H., Baxter, S.L.,

McKeown, A., Yang, G., Wu, X., Yan, F., et al.: Identifying medical diagnoses and

treatable diseases by image-based deep learning. Cell 172(5), 1122–1131 (2018)

10\. Marmolejo-Saucedo, J.A., Kose, U.: Numerical grad-cam based explainable convo-

lutional neural network for brain tumor diagnosis. Mobile Networks and Applica-

tions pp. 1–10 (2022)

11\. O’Neill, J., Delany, S.J., Mac Namee, B.: Rating by ranking: An improved scale

for judgement-based labels. In: IntRS@ RecSys. pp. 24–29 (2017)

12\. Pfeuﬀer, N., Baum, L., Stammer, W., Abdel-Karim, B.M., Schramowski, P.,

Bucher, A.M., Hügel, C., Rohde, G., Kersting, K., Hinz, O.: Explanatory inter-

active machine learning. Business & Information Systems Engineering pp. 1–25

(2023)

13\. Rahman, T., Khandakar, A., Qiblawey, Y., Tahir, A., Kiranyaz, S., Kashem,

S.B.A., Islam, M.T., Al Maadeed, S., Zughaier, S.M., Khan, M.S., et al.: Exploring

the eﬀect of image enhancement techniques on covid-19 detection using chest x-ray

images. Computers in Biology and Medicine 132, 104319 (2021)

14\. Ross, A.S., Hughes, M.C., Doshi-Velez, F.: Right for the right reasons: Train-

ing diﬀerentiable models by constraining their explanations. arXiv preprint

arXiv:1703.03717 (2017)



<a name="br11"></a> 

Unlearning Spurious Correlations in Chest X-ray Classiﬁcation

11

15\. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., Chen, L.C.: Mobilenetv2: In-

verted residuals and linear bottlenecks. In: Proceedings of the IEEE Conference

on Computer Vision and Pattern Recognition. pp. 4510–4520 (2018)

16\. Santa Cruz, B.G., Bossa, M.N., Sölter, J., Husch, A.D.: Public covid-19 x-ray

datasets and their impact on model bias–a systematic review of a signiﬁcant prob-

lem. Medical Image Analysis 74, 102225 (2021)

17\. Schramowski, P., Stammer, W., Teso, S., Brugger, A., Herbert, F., Shao, X., Luigs,

H.G., Mahlein, A.K., Kersting, K.: Making deep neural networks right for the

right scientiﬁc reasons by interacting with their explanations. Nature Machine

Intelligence 2(8), 476–486 (2020)

18\. Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., Batra, D.: Grad-

cam: Visual explanations from deep networks via gradient-based localization. In:

Proceedings of the IEEE International Conference on Computer Vision. pp. 618–

626 (2017)

19\. Shao, X., Skryagin, A., Stammer, W., Schramowski, P., Kersting, K.: Right for

better reasons: Training diﬀerentiable models by constraining their inﬂuence func-

tions. In: Proceedings of the AAAI Conference on Artiﬁcial Intelligence. vol. 35,

pp. 9533–9540 (2021)

20\. Yousefzadeh, M., Esfahanian, P., Movahed, S.M.S., Gorgin, S., Rahmati, D., Abe-

dini, A., Nadji, S.A., Haseli, S., Bakhshayesh Karam, M., Kiani, A., et al.: ai-

corona: Radiologist-assistant deep learning framework for covid-19 diagnosis in

chest ct scans. PloS One 16(5), e0250952 (2021)

21\. Zech, J.R., Badgeley, M.A., Liu, M., Costa, A.B., Titano, J.J., Oermann, E.K.:

Variable generalization performance of a deep learning model to detect pneumonia

in chest radiographs: a cross-sectional study. PLoS Medicine 15(11), e1002683

(2018)

22\. Zlateski, A., Jaroensri, R., Sharma, P., Durand, F.: On the importance of label

quality for semantic segmentation. In: Proceedings of the IEEE Conference on

Computer Vision and Pattern Recognition. pp. 1479–1487 (2018)


