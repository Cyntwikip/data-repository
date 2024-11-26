# tactiq.io free youtube transcript
# Convolutional Neural Networks Coding
# https://www.youtube.com/watch/VQrE-3tSU4U

00:00:04.040 hello everyone so in this session we'll
00:00:06.520 be doing coding for convolutional neural
00:00:11.719 network okay let's
00:00:13.920 start so um as an intro a convolutional
00:00:18.680 neural network or CNN is a type of
00:00:20.519 neural network where at least one layer
00:00:22.920 is a convolutional layer a typical CNN
00:00:25.920 architecture brings together the
00:00:27.080 following so you have a convolutional
00:00:28.640 layers to detect features pulling layers
00:00:31.039 to reduce spatial size and then Slayers
00:00:34.000 for decision making so that's your your
00:00:37.360 final layer for the output and cnns
00:00:41.399 Excel an image recognition where they've
00:00:43.600 set new benchmarks although more
00:00:45.520 recently Transformers can also um do the
00:00:49.079 same task but for the longest time it's
00:00:51.680 always CNN it's the de facto neur
00:00:54.879 Network architecture that you're going
00:00:57.000 to be using for pretty much anything
00:00:58.800 that has to do with images and
00:01:01.800 videos okay so I've mentioned earlier
00:01:04.879 convolutional layer so what's
00:01:06.600 convolution what is a convolutional
00:01:08.560 operation so a convolutional operation
00:01:11.479 is a two-step process that applies a
00:01:13.200 filter over a slice of the input
00:01:15.280 Matrix so example you have this element
00:01:18.080 wise multiplication layer between the
00:01:19.360 filter and matching section of the input
00:01:21.439 Matrix and then summation of all the
00:01:23.759 values in the resulting product Matrix
00:01:25.960 so let's consider the following example
00:01:28.040 we have a 5x5 input Matrix here see 1
00:01:31.240 to8 9 to7 and
00:01:33.680 Etc now consider that you have the
00:01:37.360 following 2x two convolutional
00:01:40.439 filter what happens is this so a
00:01:42.759 convolutional operation would slice the
00:01:44.280 filter over the different parts of the
00:01:45.640 input Matrix creating a new Matrix that
00:01:47.759 captures the patterns from the original
00:01:49.159 data in this example below the
00:01:51.079 convolution is applied to the top left
00:01:54.960 2x2 input
00:01:56.799 slice so again this is original 5x5 inut
00:02:00.039 Matrix you get the pop left
00:02:03.079 here as an example and then you simply
00:02:07.239 do an an elev wise uh multiplication
00:02:13.120 here and then you just sum it all up
00:02:17.200 right so you get 128 022 because your
00:02:22.360 convolutional filter is
00:02:24.040 one01 so it's just this and then you
00:02:26.480 just add them all up and then you end up
00:02:28.000 with 150 so that's going to be your um
00:02:31.920 value now this just one cell and you
00:02:33.959 keep on doing that for all of the slices
00:02:37.120 so we keep on just
00:02:39.040 sliding here so the next one will be
00:02:41.760 this right and then this one and then
00:02:44.040 this one and you keep on doing that for
00:02:45.480 all of the rows as well so you end up
00:02:48.720 with a new Matrix
00:02:50.560 now so let's start implementing that in
00:02:55.519 uh python so as usual here are some
00:02:58.040 preliminary um Imports that I usually do
00:03:00.720 with all of my
00:03:02.319 notebooks and we're going to be using
00:03:04.080 Tor flow and car EI for this
00:03:08.560 one so you just import it for this uh
00:03:12.440 notebook we'll be making use of um the
00:03:15.760 far than data set it consists of 60,000
00:03:18.319 colored images across 10 classes 6,000
00:03:21.360 images per class it's already split into
00:03:23.799 50k training images and 10K testing
00:03:26.040 images with no overlap between the
00:03:27.879 classes so this is how you do it and
00:03:30.599 we've also done some um normalization
00:03:33.280 because it's uh 0255 it's generally the
00:03:37.080 values that you have for colors uh but
00:03:39.599 uh we're we're normalizing it to make it
00:03:41.080 0 to
00:03:44.599 one you if you check the shape it um is
00:03:49.400 consistent with what I've said earlier
00:03:51.920 now let's do some verification of the
00:03:53.480 data so here are the 10 classes that you
00:03:55.799 have in Saar 10 data set
00:04:01.599 right here are the class names and we're
00:04:04.280 just you know going to going to show
00:04:07.239 them just for you to understand the data
00:04:09.680 better so this is uh 32 by
00:04:14.560 32
00:04:17.880 okay okay so now let's start to do the
00:04:20.600 modeling so let's build the
00:04:22.479 convolutional base the convolutional
00:04:24.479 base of our model will be a stack of con
00:04:26.759 2D and Max pooling 2D layers these
00:04:29.360 layers whole process input tensors of
00:04:30.919 shape image height width and the RGB
00:04:33.919 color channels forar that's 32 by 32 so
00:04:37.440 that's the height and width and then um
00:04:39.440 the three color channels
00:04:41.360 RGB each con 3D and Max pulling through
00:04:43.960 these layers output is a 3D tensor
00:04:46.479 height WID Channels with height with
00:04:48.840 shrinking progressively remember the
00:04:50.479 convolution operation right if you keep
00:04:51.919 on doing that it will make your Matrix
00:04:54.440 smaller so as you add more layers it
00:04:56.360 will become smaller and smaller and then
00:04:58.440 you have lots of features
00:05:03.120 and as you go deeper you generally
00:05:04.560 increase the features so that it can
00:05:07.199 capture the finer details in your data
00:05:10.320 set even as we reduce the spatial
00:05:14.759 Dimensions so this is a code for that so
00:05:17.960 you can see the con 2D layer here and
00:05:19.600 the max pooling 2D and the activation
00:05:22.240 that I use is
00:05:23.720 relu or uh
00:05:26.440 rectifying um Le leage unit it if I
00:05:30.199 remember it correctly and then the input
00:05:32.039 shape is 32 32 by
00:05:38.880 3 you have to specify that and then you
00:05:41.560 just keep on stacking them so um we use
00:05:45.120 32 64 64 uh these are arbitrary values
00:05:49.000 but generally the rule of thumb when
00:05:50.880 we're doing this is that uh we use uh
00:05:55.160 numbers that are powers of two so like
00:05:58.720 uh 2 four
00:06:00.240 816 3264 1 to8 and you get
00:06:07.560 idea oh sorry I remember now it's a
00:06:09.919 rectifying linear unit right
00:06:14.840 yeah the the leaking part is for leako I
00:06:18.160 sort of remembered that uh while I was
00:06:20.120 saying that so it's it's rectifying
00:06:21.759 linear unit there's leako which is
00:06:23.880 another activation function which
00:06:25.479 incorporates the the leakage component
00:06:28.120 there so that's why Le my
00:06:31.520 bad okay now that we have built the
00:06:36.080 convolutional model now um we're going
00:06:39.680 to add dense layers on top of it so we
00:06:42.520 do that to make the final predictions
00:06:45.000 and generally the the last uh layer that
00:06:48.319 you have corresponds to the output
00:06:52.160 classes that you have so in this case it
00:06:54.280 should end with 10 because we have 10
00:06:56.080 classes but you can add lots of add Dan
00:06:58.240 layers along the way
00:07:00.599 but the final should be 10 which is the
00:07:03.240 number of class that we have for this
00:07:05.199 data
00:07:06.479 set so we're going to take the output
00:07:08.879 from the convolutional base the shape is
00:07:11.160 4x4 by 64 if you check the model summary
00:07:14.240 here you're seeing that and we flatten
00:07:16.520 it into a 1D Vector that's the first
00:07:18.000 thing that you see here you flatten it
00:07:20.960 and then after that you add the dense
00:07:23.000 layers and it should end with 10 as I
00:07:27.120 mentioned and then after that you just
00:07:29.240 comp pilot uh you could also pay more
00:07:32.840 attention to the number of trainable
00:07:34.520 parameters here that's generally the
00:07:36.879 size of the model if you hear about like
00:07:38.599 you know um building peror model that's
00:07:41.680 that's it that's what you're seeing
00:07:44.159 here so it's a relatively lightweight
00:07:47.360 model only about 90k
00:07:50.720 parameters okay now we're compiling it
00:07:53.159 uh we use ad them Optimizer and then um
00:07:56.319 categorical cross entropy then uh we
00:07:59.479 track the accuracy metric then you just
00:08:01.400 plug in the the data set here and we're
00:08:04.159 going to have five EPO so now you're
00:08:07.039 seeing the um accuracies for the
00:08:10.319 different Epoch so we end up with about
00:08:13.599 uh
00:08:14.960 70 years um accuracy for both train and
00:08:20.599 test set which is good with such a
00:08:24.080 straightforward architecture few layers
00:08:27.599 um with minimal
00:08:29.800 um tweaking we already got 70% for this
00:08:33.599 data set if you want to see the plot
00:08:36.159 it's
00:08:38.399 this and that's it for the CNN coding
00:08:42.679 notebook
