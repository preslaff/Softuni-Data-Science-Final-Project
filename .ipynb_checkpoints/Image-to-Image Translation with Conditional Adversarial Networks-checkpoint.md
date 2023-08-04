<a name="_page0_x49.11_y71.00"></a>Image-to-Image Translation with Conditional Adversarial Networks

Phillip Isola Jun-Yan Zhu Tinghui Zhou Alexei A. Efros Berkeley AI Research (BAIR) Laboratory, UC Berkeley

fisola,junyanz,tinghuiz,efrosg@eecs.berkeley.edu

Labels to Street Scene Labels to Facade BW to Color![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.002.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.003.png)

![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.004.png) ![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.005.png)

|![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.006.png)|![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.007.png)|
| - | - |

|![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.008.png)|![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.009.png)|
| - | - |

input output Aerial to Map![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.010.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.011.png)

input output input output![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.012.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.013.png)

Day to Night Edges to Photo![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.014.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.015.png)

input output input output input output![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.016.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.017.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.018.png)

Figure 1:<a name="_page0_x85.69_y383.42"></a> Many problems in image processing, graphics, and vision involve translating an input image into a corresponding output image. These problems are often treated with application-specific algorithms, even though the setting is always the same: map pixels to pixels. Conditional adversarial nets are a general-purpose solution that appears to work well on a wide variety of these problems. Here we show results of the method on several. In each case we use the same architecture and objective, and simply train on different data.

Abstract 1. Introduction

Many problems in image processing, computer graphics, and computer vision can be posed as “translating” an input

We investigate conditional adversarial networks as a image into a corresponding output image. Just as a concept general-purpose solution to image-to-image translation may be expressed in either English or French, a scene may problems. These networks not only learn the mapping from berenderedasanRGBimage,agradientfield,anedgemap, input image to output image, but also learn a loss func- a semantic label map, etc. In analogy to automatic language tion to train this mapping. This makes it possible to apply translation, wedefineautomaticimage-to-imagetranslation

the same generic approach to problems that traditionally as the task of translating one possible representation of a arXiv:1611.07004v3  [cs.CV]  26 Nov 2018would require very different loss formulations. We demon- scene into another, given sufficienttraining data (see Figure strate that this approach is effective at synthesizing photos [1](#_page0_x85.69_y383.42)). Traditionally, each of these tasks has been tackled with from label maps, reconstructing objects from edge maps, separate, special-purpose machinery (e.g., [\[16,](#_page13_x50.11_y565.46)[ 25,](#_page13_x308.86_y194.38)[ 20,](#_page13_x50.11_y667.67)[ 9, ](#_page13_x50.11_y333.38)andcolorizingimages, amongothertasks. Indeed, sincethe [11,](#_page13_x50.11_y389.73)[ 53,](#_page14_x50.11_y520.46)[ 33,](#_page13_x308.86_y464.21)[ 39,](#_page13_x308.86_y688.50)[ 18,](#_page13_x50.11_y611.99)[ 58,](#_page14_x308.86_y72.00)[ 62](#_page14_x308.86_y162.61)]), despite the fact that the setting release of the pix2pix software associated with this pa- is always the same: predict pixels from pixels. Our goal in

per, a large number of internet users (many of them artists) this paper is to develop a common framework for all these

have posted their own experiments with our system, further problems.

demonstrating its wide applicability and ease of adoption Thecommunityhasalreadytakensignificantstepsinthis without the need for parameter tweaking. As a commu- direction, with convolutional neural nets (CNNs) becoming nity, we no longer hand-engineer our mapping functions, the common workhorse behind a wide variety of image pre- and this work suggests we can achieve reasonable results diction problems. CNNs learn to minimize a loss function – without hand-engineering our loss functions either. an objective that scores the quality of results – and although the learning process is automatic, a lot of manual effort still

1

<a name="_page1_x49.11_y71.00"></a>goes into designing effective losses. In other words, we still<a name="_page1_x308.86_y66.02"></a> x G G(x) y![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.019.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.020.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.021.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.022.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.023.png)

have to tell the CNN what we wish it to minimize. But, just ![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.024.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.025.png)

D D

like King Midas, we must be careful what we wish for! If

we take a naive approach and ask the CNN to minimize the **fake real ![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.026.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.027.png)**Euclidean distance between predicted and ground truth pix-

els, it will tend to produce blurry results [\[43,](#_page14_x50.11_y160.75)[ 62](#_page14_x308.86_y162.61)]. This is x x

because Euclidean distance is minimized by averaging all

plausible outputs, which causes blurring. Coming up with Figure 2: Training a conditional GAN to map edges!photo. The loss functions that force the CNN to do what we really want discriminator, D, learns to classify between fake (synthesized by

the generator) and real fedge, photog tuples. The generator, G,

- e.g., output sharp, realistic images – is an open problem learns to fool the discriminator. Unlike an unconditional GAN, and generally requires expert knowledge. both the generator and discriminator observe the input edge map. It would be highly desirable if we could instead specify

only a high-level goal, like “make the output indistinguish- large body of literature has considered losses of this kind, able from reality”, and then automatically learn a loss func- with methods including conditional random fields [\[10](#_page13_x50.11_y356.42)], the tion appropriate for satisfying this goal. Fortunately, this is SSIM metric [\[56](#_page14_x50.11_y632.75)], feature matching [\[15](#_page13_x50.11_y533.61)], nonparametric exactly what is done by the recently proposed Generative losses [\[37](#_page13_x308.86_y621.04)], the convolutional pseudo-prior [\[57](#_page14_x50.11_y677.82)], and losses Adversarial Networks (GANs) [\[24,](#_page13_x308.86_y160.97)[ 13,](#_page13_x50.11_y466.65)[ 44,](#_page14_x50.11_y194.53)[ 52,](#_page14_x50.11_y486.68)[ 63](#_page14_x308.86_y184.85)]. GANs based on matching covariance statistics [\[30](#_page13_x308.86_y373.98)]. The condi- learn a loss that tries to classify if the output image is real tionalGANisdifferentinthatthelossislearned,andcan, in or fake, while simultaneously training a generative model theory, penalize any possible structure that differs between to minimize this loss. Blurry images will not be tolerated output and target.

since they look obviously fake. Because GANs learn a loss Conditional GANs We are not the first to apply GANs that adapts to the data, they can be applied to a multitude of in the conditional setting. Prior and concurrent works have tasks that traditionally would require very different kinds of conditioned GANs on discrete labels [\[41,](#_page14_x50.11_y103.01)[ 23,](#_page13_x308.86_y116.64)[ 13](#_page13_x50.11_y466.65)], text [\[46](#_page14_x50.11_y262.67)], loss functions. and, indeed, images. The image-conditional models have In this paper, we explore GANs in the conditional set- tackled image prediction from a normal map [\[55](#_page14_x50.11_y599.56)], future ting. Just as GANs learn a generative model of data, condi- frame prediction [\[40](#_page14_x50.11_y72.00)], product photo generation [\[59](#_page14_x308.86_y93.19)], and tional GANs (cGANs) learn a conditional generative model image generation from sparse annotations [\[31,](#_page13_x308.86_y407.71)[ 48\]](#_page14_x50.11_y339.34) (c.f. [\[47\] ](#_page14_x50.11_y295.86)[\[24](#_page13_x308.86_y160.97)]. This makes cGANs suitable for image-to-image trans- for an autoregressive approach to the same problem). Sev- lation tasks, where we condition on an input image and gen- eral other papers have also used GANs for image-to-image erate a corresponding output image. mappings, but only applied the GAN unconditionally, re- GANs have been vigorously studied in the last two lying on other terms (such as L2 regression) to force the years and many of the techniques we explore in this pa- output to be conditioned on the input. These papers have per have been previously proposed. Nonetheless, ear- achieved impressive results on inpainting [\[43](#_page14_x50.11_y160.75)], future state lier papers have focused on specific applications, and prediction [\[64](#_page14_x308.86_y207.76)], image manipulation guided by user con- it has remained unclear how effective image-conditional straints [\[65](#_page14_x308.86_y231.35)], style transfer [\[38](#_page13_x308.86_y654.77)], and superresolution [\[36](#_page13_x308.86_y575.21)]. GANs can be as a general-purpose solution for image-to- Each of the methods was tailored for a specific applica- image translation. Our primary contribution is to demon- tion. Our framework differs in that nothing is application- strate that on a wide variety of problems, conditional specific. This makes our setup considerably simpler than GANs produce reasonable results. Our second contri- most others. bution is to present a simple framework sufficient to Our method also differs from the prior works in several achieve good results, and to analyze the effects of sev- architectural choices for the generator and discriminator. eral important architectural choices. Code is available at Unlikepastwork, forourgeneratorweusea“U-Net”-based https://github.com/phillipi/pix2pix. architecture [50], and for our [discriminator](#_page14_x50.11_y397.79) we use a convo-

lutional “PatchGAN” classifier, which only penalizes struc-

2\. Related work ture at the scale of image patches. A similar PatchGAN ar- chitecture was previously proposed in [\[38\]](#_page13_x308.86_y654.77) to capture local

Structured losses for image modeling Image-to-image style statistics. Here we show that this approach is effective translation problems are often formulated as per-pixel clas- on a wider range of problems, and we investigate the effect sification or regression (e.g., [\[39,](#_page13_x308.86_y688.50)[ 58,](#_page14_x308.86_y72.00)[ 28,](#_page13_x308.86_y284.60)[ 35,](#_page13_x308.86_y542.62)[ 62](#_page14_x308.86_y162.61)]). These of changing the patch size.

formulations treat the output space as “unstructured” in the

sense that each output pixel is considered conditionally in- 3. Method

dependent from all others given the input image. Condi-

tional GANs instead learn a structured loss. Structured GANs are generative models that learn a mapping from losses penalize the joint configuration of the output. A randomnoisevectorz tooutputimagey,G : z ! y [\[24](#_page13_x308.86_y160.97)]. In <a name="_page2_x49.11_y71.00"></a>contrast, conditional GANs learn a mapping from observed Encoder-decoder U-Net![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.028.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.029.png)

image x and random noise vector z, to y, G : fx;zg ! y.

The generator G is trained toproduce outputs that cannotbe

distinguishedfrom“real”imagesbyanadversariallytrained x y x y discriminator, D, which is trained to do as well as possible

atdetectingthegenerator’s“fakes”. Thistrainingprocedure

is diagrammed in Figure[ 2.](#_page1_x308.86_y66.02)

Figure 3: Two choices for the architecture of the generator. The

3\.1. Objective “U-Net” [\[50\]](#_page14_x50.11_y397.79) is an encoder-decoder with skip connections be- The objective of a conditional GAN can be expressed as tween mirrored layers in the encoder and decoder stacks.

LcGAN (G;D) =Ex;y [logD(x;y)]+<a name="_page2_x274.75_y213.50"></a> 3.2. Network architectures

Ex;z [log(1   D(x;G(x;z))]; (1)

We adapt our generator and discriminator architectures where G tries to minimize this objective against an ad- from those in [\[44](#_page14_x50.11_y194.53)]. Both generator and discriminator use versarial D that tries to maximize it, i.e. G = modules of the form convolution-BatchNorm-ReLu [\[29](#_page13_x308.86_y340.58)]. argminG maxD LcGAN (G;D). Details of the architecture are provided in the supplemen-

To test the importance of conditioning the discriminator, tal materials online, with key features discussed below.

we also compare to an unconditional variant in which the

discriminator does not observe x: 3.2.1 Generator with skips

LGAN (G;D) =Ey[logD(y)]+<a name="_page2_x274.75_y332.09"></a> A defining feature of image-to-image translation problems E [log(1   D(G(x;z))]: (2) isthattheymapahighresolutioninputgridtoahighresolu-

x;z tion output grid. In addition, for the problems we consider, Previous approaches have found it beneficial to mix the the input and output differ in surface appearance, but both

GAN objective with a more traditional loss, such as L2 dis- are renderings of the same underlying structure. Therefore, tance [\[43](#_page14_x50.11_y160.75)]. The discriminator’s job remains unchanged, but structure in the input is roughly aligned with structure in the the generator is tasked to not only fool the discriminator but output. We design the generator architecture around these also to be near the ground truth output in an L2 sense. We considerations.

also explore this option, using L1 distance rather than L2 as Many previous solutions [\[43,](#_page14_x50.11_y160.75)[ 55,](#_page14_x50.11_y599.56)[ 30,](#_page13_x308.86_y373.98)[ 64,](#_page14_x308.86_y207.76)[ 59\]](#_page14_x308.86_y93.19) to problems L1 encourages less blurring: in this area have used an encoder-decoder network [\[26](#_page13_x308.86_y217.82)]. In such a network, the input is passed through a series of lay-

LL1(G) = Ex;y;z [ky   G(x;z)k1]: (3) ers that progressively downsample, until a bottleneck layer, at which point the process is reversed. Such a network re-

Our finalobjective is quires that all information flow pass through all the layers, <a name="_page2_x274.75_y491.54"></a>including the bottleneck. For many image translation prob-

G= argminG maxD LcGAN (G;D) + LL1(G): (4) lems, there is a great deal of low-level information shared between the input and output, and it would be desirable to

Without z, the net could still learn a mapping from x shuttle this information directly across the net. For exam- to y, but would produce deterministic outputs, and there- ple, in the case of image colorization, the input and output fore fail to match any distribution other than a delta func- share the location of prominent edges.

tion. Past conditional GANs have acknowledged this and To give the generator a means to circumvent the bottle- provided Gaussian noise z as an input to the generator, in neck for information like this, we add skip connections, fol- addition to x (e.g., [\[55](#_page14_x50.11_y599.56)]). In initial experiments, we did not lowingthegeneralshapeofa“U-Net”[\[50](#_page14_x50.11_y397.79)]. Specifically,we find this strategy effective – the generator simply learned add skip connections between each layer i and layer n   i, to ignore the noise – which is consistent with Mathieu et where n is the total number of layers. Each skip connec- al. [\[40](#_page14_x50.11_y72.00)]. Instead, for our final models, we provide noise tion simply concatenates all channels at layer i with those only in the form of dropout, applied on several layers of our at layer n   i.

generator at both training and test time. Despite the dropout

noise, we observe only minor stochasticity in the output of 3.2.2 Markovian discriminator (PatchGAN)

our nets. Designing conditional GANs that produce highly

stochasticoutput, andtherebycapturethefullentropyofthe It is well known that the L2 loss – and L1, see Fig- conditional distributions they model, is an important ques- ure[ 4 ](#_page4_x50.11_y66.02)– produces blurry results on image generation prob- tion left open by the present work. lems [\[34](#_page13_x308.86_y508.90)]. Although these losses fail to encourage high-

**Created with an evaluation copy of Aspose.Words. To discover the full versions of our APIs please visit: https://products.aspose.com/words/**
![ref2]

frequency crispness, in many cases they nonetheless accu- rately capture the low frequencies. For problems where this is the case, we do not need an entirely new framework to enforce correctness at the low frequencies. L1 will already do.

This motivates restricting the GAN discriminator to only model high-frequency structure, relying on an L1 term to force low-frequency correctness (Eqn.[ 4](#_page2_x274.75_y491.54)). In order to model high-frequencies, it is sufficient to restrict our attention to the structure in local image patches. Therefore, we design a discriminator architecture – which we term a PatchGAN

- that only penalizes structure at the scale of patches. This discriminator tries to classify if each N N patch in an im- age is real or fake. We run this discriminator convolution- ally across the image, averaging all responses to provide the ultimate output of D.

In Section [4.4,](#_page5_x308.86_y667.91) we demonstrate that N can be much smaller than the full size of the image and still produce high quality results. This is advantageous because a smaller PatchGAN has fewer parameters, runs faster, and can be applied to arbitrarily large images.

Such a discriminator effectively models the image as a Markov random field,assuming independence between pix- els separated by more than a patch diameter. This connec- tion was previously explored in [\[38](#_page13_x308.86_y654.77)], and is also the com- mon assumption in models of texture [\[17,](#_page13_x50.11_y589.64) [21\]](#_page13_x308.86_y72.00) and style [\[16,](#_page13_x50.11_y565.46)[ 25,](#_page13_x308.86_y194.38)[ 22,](#_page13_x308.86_y93.87)[ 37](#_page13_x308.86_y621.04)]. Therefore, our PatchGAN can be under- stood as a form of texture/style loss.

3\.3. Optimization and inference

To optimize our networks, we follow the standard ap- proach from [\[24](#_page13_x308.86_y160.97)]: we alternate between one gradient de- scent step on D, then one step on G. As suggested in the original GAN paper, rather than training G to mini- mize log(1   D(x;G(x;z)), we instead train to maximize logD(x;G(x;z)) [[24\].](#_page13_x308.86_y160.97) In addition, we divide the objec- tive by 2 while optimizing D, which slows down the rate at which D learns relative to G. We use minibatch SGD and apply the Adam solver [\[32](#_page13_x308.86_y442.11)], with a learning rate of 0:0002, and momentum parameters 1 = 0:5, 2 = 0:999.

At inference time, we run the generator net in exactly the same manner as during the training phase. This differs fromtheusualprotocolinthatweapplydropoutattesttime, andweapplybatchnormalization[\[29\]](#_page13_x308.86_y340.58)usingthestatisticsof the test batch, rather than aggregated statistics of the train- ing batch. This approach to batch normalization, when the batch size is set to 1, has been termed “instance normal- ization” and has been demonstrated to be effective at im- age generation tasks [\[54](#_page14_x50.11_y564.05)]. In our experiments, we use batch sizes between 1 and 10 depending on the experiment.

<a name="_page3_x49.11_y71.00"></a>4. Experiments

To explore the generality of conditional GANs, we test themethodonavarietyoftasksanddatasets, includingboth graphics tasks, like photo generation, and vision tasks, like semantic segmentation:

- Semantic labels$photo, trained on the Cityscapes dataset [\[12](#_page13_x50.11_y422.96)].
- Architectural labels!photo, trained on CMP Facades [\[45](#_page14_x50.11_y228.31)].
- Map$aerial photo, trained on data scraped from Google Maps.
- BW!color photos, trained on [[51\].](#_page14_x50.11_y430.98)
- Edges!photo, trained on data from [[65\] ](#_page14_x308.86_y231.35)and [[60\]; ](#_page14_x308.86_y116.10)bi- naryedgesgeneratedusingtheHEDedgedetector[\[58\] ](#_page14_x308.86_y72.00)plus postprocessing.
- Sketch!photo: tests edges!photo models on human- drawn sketches from [\[19](#_page13_x50.11_y644.63)].
- Day!night, trained on [[33\].](#_page13_x308.86_y464.21)
- Thermal!color photos, trained on data from [[27\].](#_page13_x308.86_y251.20)
- Photo with missing pixels!inpainted photo, trained on Paris StreetView from [\[14](#_page13_x50.11_y499.97)].

Details of training on each of these datasets are provided in the supplemental materials online. In all cases, the in- put and output are simply 1-3 channel images. Qualita- tive results are shown in Figures[ 8,](#_page7_x50.11_y66.02)[ 9,](#_page7_x50.11_y369.44)[ 11,](#_page8_x50.11_y66.02)[ 10,](#_page7_x308.86_y369.44)[ 13,](#_page9_x50.11_y78.17)[ 14,](#_page9_x50.11_y319.16)[ 15, ](#_page10_x50.11_y78.53)[16,](#_page10_x50.11_y402.02)[ 17,](#_page11_x50.11_y75.63)[ 18,](#_page11_x50.11_y310.09)[ 19,](#_page12_x50.11_y72.55)[ 20.](#_page12_x50.11_y328.60) Several failure cases are highlighted in Figure[ 21.](#_page12_x50.11_y507.92) More comprehensive results are available at https://phillipi.github.io/pix2pix/.

Data requirements and speed We note that decent re- sults can often be obtained even on small datasets. Our fa- cade training set consists of just 400 images (see results in Figure[ 14](#_page9_x50.11_y319.16)), and the day to night training set consists of only 91 unique webcams (see results in Figure[ 15](#_page10_x50.11_y78.53)). On datasets of this size, training can be very fast: for example, the re- sultsshowninFigure[14](#_page9_x50.11_y319.16)tooklessthantwohoursoftraining on a single Pascal Titan X GPU. At test time, all models run in well under a second on this GPU.

1. Evaluation metrics

Evaluating the quality of synthesized images is an open and difficult problem [\[52](#_page14_x50.11_y486.68)]. Traditional metrics such as per- pixel mean-squared error do not assess joint statistics of the result, and therefore do not measure the very structure that structured losses aim to capture.

To more holistically evaluate the visual quality of our re- sults, we employ two tactics. First, we run “real vs. fake” perceptual studies on Amazon Mechanical Turk (AMT). For graphics problems like colorization and photo gener- ation, plausibility to a human observer is often the ultimate goal. Therefore, we test our map generation, aerial photo generation, and image colorization using this approach.

**Created with an evaluation copy of Aspose.Words. To discover the full versions of our APIs please visit: https://products.aspose.com/words/**
![ref1]

<a name="_page4_x49.11_y71.00"></a><a name="_page4_x50.11_y66.02"></a>Input Ground truth L1 cGAN L1 + cGAN

![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.031.png) ![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.032.png) ![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.033.png) ![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.034.png) ![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.035.png)

![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.036.png) ![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.037.png) ![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.038.png) ![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.039.png) ![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.040.png)

![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.041.png) ![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.042.png) ![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.043.png) ![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.044.png) ![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.045.png)

![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.046.png) ![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.047.png) ![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.048.png) ![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.049.png) ![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.050.png)

Figure 4: Different losses induce different quality of results. Each column shows results trained under a different loss. Please see https://phillipi.github.io/pix2pix/ for additional examples.

**Created with an evaluation copy of Aspose.Words. To discover the full versions of our APIs please visit: https://products.aspose.com/words/**
![ref2]

Second, we measure whether or not our synthesized cityscapesarerealisticenoughthatoff-the-shelfrecognition system can recognize the objects in them. This metric is similar to the “inception score” from [\[52](#_page14_x50.11_y486.68)], the object detec- tion evaluation in [\[55](#_page14_x50.11_y599.56)], and the “semantic interpretability” measures in [\[62\]](#_page14_x308.86_y162.61) and [\[42](#_page14_x50.11_y127.56)].

AMT perceptual studies For our AMT experiments, we followed the protocol from [\[62](#_page14_x308.86_y162.61)]: Turkers were presented with a series of trials that pitted a “real” image against a “fake” image generated by our algorithm. On each trial, each image appeared for 1 second, after which the images disappeared and Turkers were given unlimited time to re- spond as to which was fake. The first 10 images of each session were practice and Turkers were given feedback. No feedback was provided on the 40 trials of the main experi- ment. Each session tested just one algorithm at a time, and Turkers were not allowed to complete more than one ses- sion.  50 Turkers evaluated each algorithm. Unlike [\[62](#_page14_x308.86_y162.61)], we did not include vigilance trials. For our colorization ex- periments, therealandfakeimagesweregeneratedfromthe same grayscale input. For map$aerial photo, the real and fake images were not generated from the same input, in or- der to make the task more difficultand avoid floor-level re- sults. For map$aerial photo, we trained on 256256 reso-

lution images, but exploited fully-convolutional translation (described above) to test on 512  512 images, which were then downsampled and presented to Turkers at 256  256 resolution. For colorization, we trained and tested on 256  256 resolution images and presented the results to Turkers at this same resolution.

“FCN-score” While quantitative evaluation of genera- tive models is known to be challenging, recent works [\[52, ](#_page14_x50.11_y486.68)[55,](#_page14_x50.11_y599.56)[ 62,](#_page14_x308.86_y162.61)[ 42\]](#_page14_x50.11_y127.56) have tried using pre-trained semantic classifiers to measure the discriminability of the generated stimuli as a pseudo-metric. The intuition is that if the generated images are realistic, classifiers trained on real images will be able to classify the synthesized image correctly as well. To this end, we adopt the popular FCN-8s [\[39\]](#_page13_x308.86_y688.50) architecture for se- mantic segmentation, and train it on the cityscapes dataset. Wethenscoresynthesizedphotosbytheclassificationaccu- racy against the labels these photos were synthesized from.

2. Analysis of the objective function

Which components of the objective in Eqn.[ 4 ](#_page2_x274.75_y491.54)are impor- tant? We run ablation studies to isolate the effect of the L1 term, the GAN term, and to compare using a discriminator conditioned on the input (cGAN, Eqn. [1)](#_page2_x274.75_y213.50) against using an unconditional discriminator (GAN, Eqn.[ 2](#_page2_x274.75_y332.09)).

**Created with an evaluation copy of Aspose.Words. To discover the full versions of our APIs please visit: https://products.aspose.com/words/**
![ref1]

<a name="_page5_x49.11_y71.00"></a><a name="_page5_x50.11_y66.02"></a>L1 L1+cGAN that the output look realistic. This variant results in poor performance;![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.051.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.052.png) examining the results reveals that the gener-

ator collapsed into producing nearly the exact same output regardless of input photograph. Clearly, it is important, in

this case, that the loss measure the quality of the match be- tween input and output, and indeed cGAN performs much ![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.053.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.054.png)better than GAN. Note, however, that adding an L1 term also encourages that the output respect the input, since the L1 loss penalizes the distance between ground truth out- puts, which correctly match the input, and synthesized out- puts, which may not. Correspondingly, L1+GAN is also

Figure 5: Adding skip connections to an encoder-decoder to create effective at creating realistic renderings that respect the in- <a name="_page5_x50.11_y218.35"></a>a “U-Net” results in much higher quality results. put label maps. Combining all terms, L1+cGAN, performs

Loss Per-pixel acc. Per-class acc. Class IOU similarly well.

L1 0.42 0.15 0.11 Colorfulness A striking effect of conditional GANs is GAN 0.22 0.05 0.01 that they produce sharp images, hallucinating spatial struc- cGAN 0.57 0.22 0.16

L1+GAN 0.64 0.20 0.15 ture even where it does not exist in the input label map. One L1+cGAN 0.66 0.23 0.17 mightimaginecGANshaveasimilareffecton“sharpening”

Ground truth 0.80 0.26 0.21 in the spectral dimension – i.e. making images more color- Table 1: FCN-scores for different losses, evaluated on Cityscapes ful. Just as L1 will incentivize a blur when it is uncertain <a name="_page5_x50.11_y322.36"></a>labels$photos. where exactly to locate an edge, it will also incentivize an

average, grayish color when it is uncertain which of sev- Loss Per-pixel acc. Per-class acc. Class IOU eral plausible color values a pixel should take on. Specially,

EncoderEncoder-decoder-decoder (L1)(L1+cGAN) 0.350.29 0.120.09 0.080.05 L1 will be minimized by choosing the median of the condi- U-net (L1) 0.48 0.18 0.13 tional probability density function over possible colors. An

U-net (L1+cGAN) 0.55 0.20 0.14 adversarial loss, on the other hand, can in principle become Table 2: FCN-scores for different generator architectures (and ob- aware that grayish outputs are unrealistic, and encourage jectives), evaluated on Cityscapes labels$photos. (U-net (L1- matching the true color distribution [[24\].](#_page13_x308.86_y160.97) In Figure [7, ](#_page6_x50.11_y200.89)we

cGAN)scoresdifferfromthosereportedinothertablessincebatch investigate whether our cGANs actually achieve this effect size was 10 for this experiment and 1 for other tables, and random on the Cityscapes dataset. The plots show the marginal dis- <a name="_page5_x50.11_y442.01"></a>variation between training runs.) tributions over output color values in Lab color space. The

Discriminator ground truth distributions are shown with a dotted line. It receptive field Per-pixel acc. Per-class acc. Class IOU is apparent that L1 leads to a narrower distribution than the

116116 0.390.65 0.150.21 0.170.10 ground truth, confirmingthe hypothesis that L1 encourages 7070 0.66 0.23 0.17 average, grayish colors. Using a cGAN, on the other hand,

286286 0.42 0.16 0.11 pushes the output distribution closer to the ground truth.

Table 3: FCN-scores for different receptive field sizes of the dis-

criminator, evaluated on Cityscapes labels!photos. Note that in- 4.3.<a name="_page5_x308.86_y515.63"></a> Analysis of the generator architecture

put images are 256  256 pixels and larger receptive fields are

padded with zeros. A U-Net architecture allows low-level information to shortcutacrossthenetwork. Doesthisleadto better results?

Figure[ 4 ](#_page4_x50.11_y66.02)shows the qualitative effects of these variations Figure[5](#_page5_x50.11_y66.02)andTable[2](#_page5_x50.11_y322.36)comparetheU-Netagainstanencoder-

on two labels!photo problems. L1 alone leads to reason- decoder on cityscape generation. The encoder-decoder is able but blurry results. The cGAN alone (setting = 0 in created simply by severing the skip connections in the U- Eqn.[ 4)](#_page2_x274.75_y491.54) gives much sharper results but introduces visual ar- Net. The encoder-decoder is unable to learn to generate tifacts on certain applications. Adding both terms together realistic images in our experiments. The advantages of the (with = 100) reduces these artifacts. U-Net appear not to be specificto conditional GANs: when We quantify these observations using the FCN-score on bothU-Netandencoder-decoderaretrainedwithanL1loss,

thecityscapeslabels!phototask(Table1[): ](#_page5_x50.11_y218.35)theGAN-based the U-Net again achieves the superior results.

objectives achieve higher scores, indicating that the synthe-

sized images include more recognizable structure. We also 4.4.<a name="_page5_x308.86_y667.91"></a>FromPixelGANstoPatchGANstoImageGANs test the effect of removing conditioning from the discrimi-

nator (labeled as GAN). In this case, the loss does not pe- We test the effect of varying the patch size N of our dis- nalize mismatch between the input and output; it only cares criminator receptive fields, from a 1  1 “PixelGAN” to a

<a name="_page6_x49.11_y71.00"></a><a name="_page6_x50.11_y66.02"></a>L1 11 1616 7070 286286![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.055.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.056.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.057.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.058.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.059.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.060.png)

Figure 6: Patch size variations. Uncertainty in the output manifests itself differently for different loss functions. Uncertain regions become blurry and desaturated under L1. The 1x1 PixelGAN encourages greater color diversity but has no effect on spatial statistics. The 16x16 PatchGAN creates locally sharp results, but also leads to tiling artifacts beyond the scale it can observe. The 7070 PatchGAN forces outputs that are sharp, even if incorrect, in both the spatial and spectral (colorfulness) dimensions. The full 286286 ImageGAN produces results that are visually similar to the 7070 PatchGAN, but somewhat lower quality according to our FCN-score metric (Table[ 3](#_page5_x50.11_y442.01)). Please <a name="_page6_x50.11_y200.89"></a>see https://phillipi.github.io/pix2pix/ for additional examples.

−1 −1 −1

−3 −3 −3 Histogram intersection −5 −5 −5 against ground truth

−7 −7 −7 Loss L a b

−9 L1cGAN −9 −9 L1 0.81 0.69 0.70 L1+cGAN cGAN 0.87 0.74 0.84

−110 20 40 60 L1+pixelcGAN~~Ground truth~~80 100 −1170 90 110 130 −11 L1+cGAN 0.86 0.84 0.82 L a 70 90 110b 130 150 PixelGAN 0.83 0.68 0.78

(a) (b) (c) (d)

Figure 7: Color distribution matching property of the cGAN, tested on Cityscapes. (c.f. Figure 1 of the original GAN paper [\[24](#_page13_x308.86_y160.97)]). Note that the histogram intersection scores are dominated by differences in the high probability region, which are imperceptible in the plots, which show log probability and therefore emphasize differences in the low probability regions.

full 286  286 “ImageGAN”[1.](#_page6_x64.46_y682.61) Figure[ 6 ](#_page6_x50.11_y66.02)shows qualitative<a name="_page6_x308.86_y364.77"></a> Photo ! Map Map ! Photo

results of this analysis and Table[ 3 ](#_page5_x50.11_y442.01)quantifiesthe effects us- L1Loss 2.8%  1.0% 0.8%  0.3%

% Turkers labeled real % Turkers labeled real

ing the FCN-score. Note that elsewhere in this paper, unless L1+cGAN 6.1%  1.3% 18.9%  2.5% specified, all experiments use 70  70 PatchGANs, and for<a name="_page6_x308.86_y421.36"></a> Table 4: AMT “real vs fake” test on maps$aerial photos.

this section all experiments use an L1+cGAN loss.

The PixelGAN has no effect on spatial sharpness but MethodL2 regression from [[62\]](#_page14_x308.86_y162.61) % T16.3%urkers labeled 2.4%real

does increase the colorfulness of the results (quantified in Zhang et al. 2016 [[62\]](#_page14_x308.86_y162.61) 27.8%  2.7%

Figure[ 7](#_page6_x50.11_y200.89)). For example, the bus in Figure[ 6 ](#_page6_x50.11_y66.02)is painted gray Ours 22.5%  1.6%

when the net is trained with an L1 loss, but becomes red Table 5: AMT “real vs fake” test on colorization.

with the PixelGAN loss. Color histogram matching is a

commonprobleminimageprocessing[\[49](#_page14_x50.11_y364.10)], andPixelGANs generator convolutionally, on larger images than those on may be a promising lightweight solution. which it was trained. We test this on the map$aerial photo

Using a 1616 PatchGAN is sufficientto promote sharp task. Aftertrainingageneratoron256256images, wetest outputs, and achieves good FCN-scores, but also leads to it on 512512 images. The results in Figure[ 8 ](#_page7_x50.11_y66.02)demonstrate tiling artifacts. The 70  70 PatchGAN alleviates these the effectiveness of this approach.

artifacts and achieves slightly better scores. Scaling be-

yond this, to the full 286  286 ImageGAN, does not ap- 4.5. Perceptual validation

pear to improve the visual quality of the results, and in

We validate the perceptual realism of our results on the fact gets a considerably lower FCN-score (Table[ 3](#_page5_x50.11_y442.01)). This

tasks of map$aerial photograph and grayscale!color. Re- may be because the ImageGAN has many more parameters

sults of our AMT experiment for map$photo are given in and greater depth than the 70  70 PatchGAN, and may be

Table[ 4.](#_page6_x308.86_y364.77) The aerial photos generated by our method fooled harder to train.

Fully-convolutional translation An advantage of the participantsbaseline, whichon 18produces:9% of trials,blurrysignificantlyresults andabonearlyve thenevL1er PatchGAN is that a fixed-size patch discriminator can be

applied to arbitrarily large images. We may also apply the fooledtion ourparticipants.method onlyInfooledcontrast,participantsin the photoon!6:map1% ofdirec-tri- 1<a name="_page6_x64.46_y682.61"></a> als, and this was not significantly different than the perfor-![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.061.png)

GANWdiscriminatore achieve this. Detailsvariationof thisin patchprocess,sizeandby theadjustingdiscriminatorthe deptharchitec-of the mance of the L1 baseline (based on bootstrap test). This tures, are provided in the in the supplemental materials online. may be because minor structural errors are more visible

Map to aerial photo Aerial photo to map



|![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.062.png)|![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.063.png)|
| - | - |

|![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.064.png)|![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.065.png)|
| - | - |
|![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.066.png)|![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.067.png)|

|![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.068.png)|![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.069.png)|
| - | - |

input output input output

<a name="_page7_x49.11_y71.00"></a><a name="_page7_x50.11_y66.02"></a>Figure 8: Example results on Google Maps at 512x512 resolution (model was trained on images at 256  256 resolution, and run convo- <a name="_page7_x50.11_y369.44"></a>lutionally on the larger images at test time). Contrast adjusted for clarity<a name="_page7_x308.86_y369.44"></a>.

[Classification](#_page14_x308.86_y162.61) Ours Input Ground truth L1 cGAN L2 ![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.070.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.071.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.072.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.073.png)[\[62\]](#_page14_x308.86_y162.61) (rebal.) [\[62\]](#_page14_x308.86_y162.61) (L1 + cGAN) Ground truth

Figure 10: Applying a conditional GAN to semantic segmenta- tion.![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.074.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.075.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.076.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.077.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.078.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.079.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.080.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.081.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.082.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.083.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.084.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.085.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.086.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.087.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.088.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.089.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.090.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.091.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.092.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.093.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.094.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.095.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.096.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.097.png) The cGAN produces sharp images that look at glance like the ground truth, but in fact include many small, hallucinated ob- jects.

ble[ 5](#_page6_x308.86_y421.36)). We also tested the results of [\[62\]](#_page14_x308.86_y162.61) and a variant of their method that used an L2 loss (see [\[62\]](#_page14_x308.86_y162.61) for details). The

Figure 9: Colorization results of conditional GANs versus the L2 conditional GAN scored similarly to the L2 variant of [\[62\] ](#_page14_x308.86_y162.61)regression from [\[62\]](#_page14_x308.86_y162.61) and the full method (classification with re- (difference insignificant by bootstrap test), but fell short of balancing) from [\[64](#_page14_x308.86_y207.76)]. The cGANs can produce compelling col- [\[62](#_page14_x308.86_y162.61)]’s full method, which fooled participants on 27:8% of orizations (first two rows), but have a common failure mode of trials in our experiment. We note that their method was producing a grayscale or desaturated result (last row).

specificallyengineered to do well on colorization.

in maps, which have rigid geometry, than in aerial pho- 4.6. Semantic segmentation

tographs, which are more chaotic.

We trained colorization on ImageNet [\[51](#_page14_x50.11_y430.98)], and tested Conditional GANs appear to be effective on problems on the test split introduced by [\[62,](#_page14_x308.86_y162.61)[ 35](#_page13_x308.86_y542.62)]. Our method, with where the output is highly detailed or photographic, as is L1+cGAN loss, fooled participants on 22:5% of trials (Ta- common in image processing and graphics tasks. What

**Created with an evaluation copy of Aspose.Words. To discover the full versions of our APIs please visit: https://products.aspose.com/words/**
![ref3]

![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.099.png) ![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.100.png) ![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.101.png)

*by Kaihu Chen by Jack Qiao by Mario Klingemann![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.102.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.103.png)*

#fotogenerator![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.104.png)![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.105.png)

*sketch by Ivy Tsai by Bertrand Gondouin by Brannon Dorsey sketch by Yann LeCun![](Image-to-Image%20Translation%20with%20Conditional%20Adversarial%20Networks.106.png)*

**This document was truncated here because it was created in the Evaluation Mode.**
**Created with an evaluation copy of Aspose.Words. To discover the full versions of our APIs please visit: https://products.aspose.com/words/**

[ref1]: <Image-to-Image Translation with Conditional Adversarial Networks.001.png>
[ref2]: <Image-to-Image Translation with Conditional Adversarial Networks.030.png>
[ref3]: <Image-to-Image Translation with Conditional Adversarial Networks.098.png>
