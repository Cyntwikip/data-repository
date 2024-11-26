# tactiq.io free youtube transcript
# Transfer Learning Coding
# https://www.youtube.com/watch/d2hLvqXqVsk

00:00:02.879 hello everyone so in this session we'll
00:00:05.400 be doing some coding about transfer
00:00:08.760 learning we'll explore transfer learning
00:00:11.360 first and then uh we'll train um a
00:00:14.360 neural network model from scratch and
00:00:16.320 then we'll see how using a pre-trained
00:00:19.000 model can significantly boost
00:00:20.400 performance and as usual in my notebooks
00:00:22.760 um there's some preliminary soell PR
00:00:25.000 importing some stuff now again we're
00:00:28.640 going to be building our simple CNN
00:00:30.800 model for comparison purposes so that
00:00:33.200 you appreciate transfer your learning
00:00:35.280 I'll skip the detailed explanation since
00:00:36.840 I've already covered this in a previous
00:00:38.480 lecture and notebook you may refer to
00:00:40.440 that to keep training and INF sufficient
00:00:43.399 I'll only be using 5K samples for
00:00:45.520 training and onek for testing still a
00:00:48.079 good solid uh sample size for our
00:00:51.399 purposes
00:00:53.920 here so um I'm going to be preparing the
00:00:57.399 data not going to explain too about it
00:01:00.559 but uh at a high level I'm going to be
00:01:02.960 setting the seed so that our um results
00:01:06.000 are consistent I'm going to be making
00:01:07.720 use of cy1 data set uh which I've also
00:01:10.799 used in the previous lecture and
00:01:13.280 notebook about uh
00:01:15.159 CNN and then I'm going to extract the
00:01:17.439 random 1ky samples as I mentioned and
00:01:20.040 then some transformation here make it um
00:01:22.880 from0 to 55 to um 0 to1 scale instead
00:01:29.000 and then I'm just going to convert the
00:01:30.439 class vectors to Binary class matrices
00:01:33.119 here or I mean multiclass sorry it's a
00:01:36.759 multiclass um output here because it's
00:01:40.720 uh Cy 10 there is
00:01:45.200 10
00:01:47.759 yeah then if you check the M Max values
00:01:51.000 you have 0 to one now let's build the
00:01:54.280 model um and I won't be discussing too
00:01:56.759 much about this because this is the same
00:01:58.360 that we have in the CNN
00:02:02.680 mod so if you check here our scores
00:02:08.520 are um about 62k uh 62% for the train
00:02:14.480 set and the Val set or Che test set is
00:02:18.319 about
00:02:19.280 52% right that's our comparison it's all
00:02:21.800 the same data set that we're going to be
00:02:23.239 using later
00:02:25.319 on okay intro to transfer learning
00:02:29.160 transfer learning leverages features
00:02:30.720 from a model train on one problem to
00:02:32.480 tackle a related task like using cat PR
00:02:34.800 detection model to help identify other F
00:02:37.519 creatures such as lion and tiger this
00:02:40.280 approach is especially useful when data
00:02:41.800 is limited making training full model
00:02:44.280 from scratch
00:02:46.040 impractical so here's the transfer
00:02:48.760 learning
00:02:49.840 workflow typical workflow and deep
00:02:52.040 learning transfer learning involves
00:02:54.280 we're using a pre-train model here we
00:02:56.920 import layers and weights from an
00:02:58.120 already trained model and I'm going to
00:02:59.720 be showing an example of that later uh
00:03:02.400 we're going to be freezing the layers we
00:03:04.640 lock these layers to preserve their
00:03:05.879 learn information essentially the
00:03:07.519 previous weights we um still utilize the
00:03:12.319 trained model we set trainable equals to
00:03:15.480 false you'll see that later um next is
00:03:18.599 uh we add new layers we stack new
00:03:20.560 chainable layers on top of the base
00:03:21.799 model that we use to adapt to the
00:03:25.319 existing
00:03:26.200 features uh sorry we use the newr layers
00:03:30.959 so that we adapt the existing features
00:03:33.360 to the predictions on our specific data
00:03:35.920 set because it's a new data set a new
00:03:37.840 example now we have to do some
00:03:39.840 customization so this is where it comes
00:03:42.000 from um this is optional the next part
00:03:44.480 it's fine tuning we infree part or all
00:03:47.480 of the model and training it on new data
00:03:49.640 with a low learning rate this can
00:03:51.400 further refine features enhancing
00:03:52.959 performance for the new task and lastly
00:03:55.720 um we of course now train it on the new
00:03:58.319 data after we've done all of these
00:04:00.319 things right um we want to train it um
00:04:03.640 to our new data
00:04:06.599 set now let's build our base
00:04:10.680 model and for this one I'm going to be
00:04:13.000 making use of mobilet V2 it is a deep
00:04:15.599 neural network architecture designed by
00:04:17.079 Google for efficient mobile and Edge
00:04:19.320 device applications it builds on its
00:04:21.600 predecessor mobile net V1 obviously by
00:04:24.759 improving both speed and accuracy while
00:04:26.240 minimizing computational costs making it
00:04:28.400 ideal for devices with limited resources
00:04:30.440 like smartphones and iot
00:04:32.520 devices I'll be using this model and
00:04:35.080 this notebook because it's very
00:04:36.720 lightweight and Google cab oror memory
00:04:39.280 limit can handle it normally I'll be
00:04:41.320 making use of of my own laptop and I'm
00:04:44.160 going to utilize a bigger model but you
00:04:46.479 know um since I want to make this more
00:04:49.600 accessible to more people I'm trying to
00:04:52.080 use collab and let's see what I can um
00:04:54.919 squeeze out of it so uh with lots of
00:04:58.000 testing this is one of the model
00:05:00.440 that I can utilize that works for the
00:05:03.280 specific data set with the sampling and
00:05:06.280 all of
00:05:07.479 that anyway um this is one of the models
00:05:11.560 um that are available in Caris one of
00:05:14.720 the pre-train models out there normally
00:05:17.240 again I'd make use of more powerful ones
00:05:20.199 so this is how you import it very simple
00:05:22.319 right you already have it
00:05:24.000 here so you can see that um it has how
00:05:28.360 many weights
00:05:31.560 uh lots of
00:05:32.759 scrolling so you have total parameter
00:05:37.600 size of about 2 million right that's a
00:05:40.680 model
00:05:43.520 size now um we're going to be doing
00:05:47.039 transfer learning
00:05:48.840 here uh of course we have to do some
00:05:51.039 data
00:05:51.880 pre-processing so we will be resizing
00:05:54.759 our input images because mobile net V2
00:05:58.440 works on
00:06:00.280 images that have higher resolution so
00:06:03.720 initially our data set the sr10 data set
00:06:06.599 is 32x 32 by three but I I'll just scale
00:06:10.400 it up so from 32 I'll make it 96 so this
00:06:13.479 is how you do
00:06:15.360 it now um I'm going to be using the same
00:06:19.080 pre-processing that they've done in the
00:06:20.919 original model so this is how we use
00:06:23.319 that uh use the pre-process input it's
00:06:25.560 one of the things that we have imported
00:06:26.960 as
00:06:27.680 well if you check the minmax it's
00:06:29.759 negative 1 to1 so that must be the
00:06:31.400 scaling that they have done in the in
00:06:33.280 the original uh
00:06:36.680 model okay now second step we freeze
00:06:39.880 layers and um Pine tuning is is optional
00:06:43.160 as I
00:06:44.120 mentioned so the layers in a neural
00:06:46.560 network model have two types of Weights
00:06:48.560 you have the trainable and non-trainable
00:06:50.319 trainable as the name implies uh you
00:06:53.240 train it when when you uh when you chain
00:06:55.319 the model the those weights get
00:06:57.680 updated to minimize the
00:07:00.400 whereas non-trainable weights there are
00:07:02.800 there are also parameters that you have
00:07:04.360 in the model but you don't train them
00:07:06.560 don't update them during forward pass
00:07:09.160 those weights are used as this could
00:07:12.560 also call these attributes and car just
00:07:14.919 to um double
00:07:17.039 check I'm also including the code
00:07:20.000 snippet for doing Pine tuning if you
00:07:23.080 want to um retrain specific layers or
00:07:27.240 freeze or unfreeze certain parts this is
00:07:29.319 this this is what you
00:07:31.240 do uh you just replace with the target
00:07:33.759 layer's name so that um by doing this
00:07:37.120 those those layers are
00:07:38.840 updated note that uh first thing that
00:07:41.759 you do here is to set the trainable to
00:07:44.479 equals false that means the whole base
00:07:46.560 model is not trainable and then after
00:07:48.319 that we're
00:07:49.599 manually um making some of the layers
00:07:52.400 trainable that's what's happening here
00:07:55.360 but uh we're going to skip fine tuning
00:07:57.000 here um we're just going to freeze the
00:07:59.039 base model asses and we're going to
00:08:00.560 stack the the den
00:08:04.280 layers again so we have added a few
00:08:06.520 layers after the base model so that it
00:08:08.120 can adapt to our new data set so again
00:08:10.639 I've set the do trainable attribute to
00:08:15.360 false and then I've added the flatten
00:08:18.919 some Dropout layer um you could remove
00:08:21.479 that but generally drop out is good you
00:08:24.479 can tweak the values if you want now I'm
00:08:26.919 adding the dense layers here so there's
00:08:29.520 1 to8 64 and then finally
00:08:32.958 the uh 10 the last layer is 10 of course
00:08:36.559 because you have 10 classes and the
00:08:38.519 activation is soft
00:08:40.360 Max so if you check here the total
00:08:44.000 number of parameters Now by adding um
00:08:48.000 the dense layers would be 3.7 million
00:08:51.120 and the trainable parameters is 1.4 the
00:08:54.160 non-t trainables
00:08:55.640 are about 2.2
00:08:58.279 million now now uh you compile it after
00:09:02.120 you've done preparing the model you
00:09:04.560 compile it so 10 epox as well same data
00:09:07.720 set but uh we had to scale it up now uh
00:09:11.200 you can see that still very fast right
00:09:12.800 if you compare the the training time
00:09:16.279 with our previous CNN model you'll see
00:09:18.000 that this is also quite fast even though
00:09:20.000 we're making use of a bigger model and
00:09:22.079 then at the end of it we got 99%
00:09:25.320 accuracy for the train set and 83% accur
00:09:29.640 for
00:09:30.920 the which are much higher than the
00:09:33.880 manually train CNN model I'll I'll
00:09:36.160 scroll up a bit so you can see uh the
00:09:39.360 the time it took for for us to train the
00:09:41.120 CNN model of course it's a small model
00:09:44.040 so it's very fast right but now um see
00:09:48.399 the
00:09:49.399 the the size difference from like a very
00:09:52.680 small CNN model like about 90k
00:09:56.079 parameters we now have like um three
00:09:58.560 million
00:10:00.279 um in terms of size in terms of
00:10:02.480 parameters but uh it's still quite fast
00:10:04.920 because we're not actually training the
00:10:07.200 the base model we're training the
00:10:09.279 additional dense
00:10:11.720 layers which is also heavy right because
00:10:14.160 of the 128 and 64 and 10 which we see
00:10:16.680 here but uh with just with just few
00:10:19.880 modifications we're already getting very
00:10:22.320 good performances out of it and that's
00:10:24.000 the power of transfer
00:10:25.760 learning so if you want to see the graph
00:10:27.959 here you see um this right quite stable
00:10:31.920 right uh in the first depac we already
00:10:34.640 got very good test and train accuracies
00:10:40.120 apparently and um that's it for transfer
00:10:42.600 learning
