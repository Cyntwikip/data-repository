# tactiq.io free youtube transcript
# Model Interpretability - Model-Agnostic Methods Coding
# https://www.youtube.com/watch/OZauYbIvYew

00:00:04.920 hello everyone so for this session I'll
00:00:07.520 be um discussing the implementation for
00:00:10.280 model agnostic methods so as usual I'll
00:00:12.639 be importing the
00:00:14.200 libraries and then I'll be preparing the
00:00:16.320 data set which in this case is the
00:00:17.880 Titanic data set and then uh I'll be
00:00:20.720 doing some of the pre-processing such as
00:00:22.439 uh imputation and train test
00:00:24.920 split so there are two approaches uh for
00:00:27.400 doing model agnostic methods there's the
00:00:29.039 global and the local one let's start
00:00:30.920 with the global model agnostic methods
00:00:33.200 and again we use that for models that
00:00:35.000 are not inherently
00:00:38.719 interpretable so one approach for Global
00:00:41.840 model agnostic methods
00:00:47.120 is answers the question what features
00:00:49.520 have the biggest impact on predictions
00:00:51.680 permutation importance is a method for
00:00:53.120 measuring the impact of a feature on
00:00:54.239 models predictions this is done by
00:00:55.480 randomly shuffling One features values
00:00:57.600 and observing how much this disrupts
00:00:59.440 models per
00:01:01.079 performance this is a more specific
00:01:03.440 example of sensitivity analysis and
00:01:05.199 that's normally the term that we use in
00:01:06.920 other fields but uh in the context of
00:01:09.400 machine learning um it's permutation
00:01:11.520 importance but if you hear sensitivity
00:01:13.080 analysis out there it's the same thing
00:01:15.720 so the process for that is you change
00:01:17.720 the model whatever model that you have
00:01:19.360 and then you Shuffle the features values
00:01:21.280 and measure the drop in performance and
00:01:23.720 then you restore the original features
00:01:25.159 values and repeat the process for other
00:01:26.880 features that you have
00:01:30.439 the more a model's performance drops
00:01:32.079 after shuffling a feature the more
00:01:33.720 important that feature is so in this
00:01:36.280 case I I'll be I'll be using random
00:01:38.360 Force classifier then I'll train the
00:01:40.320 model and then you can see the score
00:01:42.200 here and then I'll apply feature
00:01:44.560 importance good thing there's uh a
00:01:47.960 function in um pyit learn for feature
00:01:50.840 importance so you could plug in the
00:01:53.119 model here and then you can see the
00:01:55.240 importance of the different features so
00:01:59.039 in this case since this is the Titanic
00:02:00.920 data set you'd expect that it's the sex
00:02:03.039 that um has the highest um
00:02:08.399 predictor um component in the in this
00:02:11.480 model so if you have a high sex male
00:02:14.400 value which in this case is um a male
00:02:17.879 person then that means you have uh you
00:02:21.480 have the most importance and that
00:02:24.080 actually decreases
00:02:25.920 the um survivability of the specific
00:02:29.519 data
00:02:30.519 point so this uh this is the
00:02:33.280 visualization for that you see the X
00:02:34.800 male P class fair having the highest
00:02:36.959 importance you could also see the
00:02:39.519 spread so this is the decrease in
00:02:41.800 accuracy
00:02:45.280 score so how to interpret that yeah as
00:02:48.879 mentioned the most important features
00:02:50.440 appear at the top the values reflect how
00:02:52.720 much shuffling each feature affects
00:02:54.239 model's
00:02:55.879 performance in some cases negative
00:02:58.080 values will occur but that's due to
00:03:02.360 Randomness and um that happens often in
00:03:06.200 small data sets due to
00:03:08.159 chance another approach for Global model
00:03:10.680 agnostic method is partial dependence
00:03:12.840 plots um it's for visual visualizing the
00:03:16.280 relationship between a feature and the
00:03:17.799 target as usual we hold all other
00:03:20.280 features constant they help answer
00:03:22.680 questions like how this changing one
00:03:24.040 feature while keeping others the same
00:03:25.480 affect the
00:03:26.560 prediction so the process for that I
00:03:28.680 would say is quite simil to um hyper
00:03:31.360 parameter
00:03:33.000 tuning so you select the feature first
00:03:36.439 um that you want to analyze and then
00:03:37.959 create a value grid like hyper parameter
00:03:39.920 tuning generate a list of values range
00:03:42.159 of values for that feature and then uh
00:03:45.519 you make
00:03:46.799 predictions based on the different uh
00:03:50.840 values for that specific feature and You
00:03:52.760 observe the
00:03:54.400 performance the changes in performance
00:03:56.519 for that specific feature and then you
00:03:59.200 average the predictions and then you
00:04:00.760 visualize and interpret so you could do
00:04:03.159 that as well using SK SK learn and you
00:04:05.519 could see the different um impact on uh
00:04:10.079 performance in uh the age feature so you
00:04:14.000 can see that it generally drops if you
00:04:15.920 have a higher
00:04:19.000 age could also do it for
00:04:22.800 2D so instead of just one feature like
00:04:26.040 uh age could have uh age and then
00:04:30.600 sucks so you could see the impact here
00:04:33.280 in this case the um the in the graph
00:04:37.720 above the lighter U indicates better
00:04:40.120 performance we can see that if the
00:04:42.120 passenger is female not sex male and the
00:04:44.840 age is lower then you have the highest
00:04:47.440 chances of
00:04:49.199 survival could try it for other features
00:04:52.000 as well like sax Mill and fair same idea
00:04:55.600 passengers that have higher fair and are
00:04:57.280 female have higher chances of survival
00:05:02.039 this part is not so
00:05:04.080 obvious um so in some cases it's quite
00:05:06.440 tricky to use that but um the same
00:05:08.280 concept still applies so in this case
00:05:09.759 it's fair and age again it's not a
00:05:11.840 straightforward but generally the
00:05:13.000 passengers that have higher fair and
00:05:14.199 lower age are likely to
00:05:16.120 survive now let's proceed to local model
00:05:18.600 agnostic
00:05:20.120 methods so we could use shop for
00:05:23.240 that it's it answers this question so
00:05:26.120 what if you want to break down how the
00:05:28.080 model works for an individual prediction
00:05:30.639 so shap or shly additive explanations
00:05:32.720 values provide local explanations for
00:05:34.840 individual predictions by showing the
00:05:36.319 contribution of each feature to a
00:05:37.720 model's
00:05:38.960 output so you're going to have to make
00:05:41.120 use of another library for
00:05:44.280 that for um demonstration purposes let's
00:05:48.680 just use one data point so in this case
00:05:50.880 it's
00:05:52.000 five right and then let's predict the
00:05:56.080 the probability of survival
00:06:00.319 now let's use shop so you've seen
00:06:02.080 earlier that it's 96.25% chance of
00:06:04.319 survival but
00:06:06.680 why so we could use free explainer here
00:06:10.120 because it's a tree based
00:06:12.440 model I'm going to show other uh ways of
00:06:16.080 uh doing explanation later but for now
00:06:17.880 it's free
00:06:19.440 explainer and
00:06:21.759 then you use shap so it's just two lines
00:06:25.080 here you plug in the values and um it
00:06:28.080 shows you the impact of different
00:06:30.599 features to our output which in this
00:06:34.319 case is 96% You' also see the base value
00:06:37.120 here which is between 0.3 and
00:06:40.240 0.4 so uh because of these features we
00:06:45.000 got the 96% so
00:06:48.400 um lower sex male value which is zero
00:06:51.520 that means female um increased the
00:06:53.919 chances of survival by a lot then
00:06:55.759 followed by uh P class here then fair
00:06:58.599 then um low age quite young there's also
00:07:02.520 theas
00:07:03.960 which decreased
00:07:06.759 survival just by a little
00:07:09.360 bit but basically with all these
00:07:11.360 features combined you have the
00:07:16.400 96% again as mentioned earlier the top
00:07:18.960 features are sex male pclass and
00:07:22.440 fair and that's uh those are the main
00:07:25.199 reasons why we got that survival rate
00:07:30.440 now we could also do that for multiple
00:07:31.759 data points and this is the code for
00:07:33.879 that um you might have to use a
00:07:37.680 different view for you to be able to
00:07:39.479 visualize it
00:07:40.840 properly um and uh if you run it using
00:07:43.720 your notebook you could you could um
00:07:46.560 better see this I I used my personal
00:07:49.639 notebook for this so if you use collab
00:07:52.039 you will see this but if you run us in
00:07:54.680 your notebook you will see this and also
00:07:57.520 it would be better if you use um light
00:07:59.680 mode for that specific
00:08:05.960 Vis now what if you want to have an
00:08:08.800 overall view for um the model instead of
00:08:12.759 just a specific data point you want to
00:08:16.199 you want the whole data set so you could
00:08:17.720 also do that using shap summary plot so
00:08:20.639 you could see the visualization here and
00:08:23.319 uh there's a guide to interpret this so
00:08:25.560 the vertical position indicates which
00:08:27.000 feature it represents
00:08:29.960 right so This One X MP class Fair the
00:08:33.039 color shows whether the features value
00:08:34.839 is high or low for the given Row in the
00:08:36.599 data set so it's this
00:08:39.159 one color here if it's um pink reddish
00:08:43.958 that means it's high if it's blue that's
00:08:46.519 that's low and then the horizontal
00:08:48.160 position shows whether the features
00:08:50.120 value contributes to a higher l or lower
00:08:53.160 prediction so generally um if you have a
00:08:56.959 high um value for sex male that means
00:09:00.279 male then you have lower chances of
00:09:04.160 survival as you can see here cuz it's
00:09:05.600 negative sha value and vice versa and
00:09:08.480 same goes for other features could also
00:09:11.120 simply get the average of the impact
00:09:13.120 instead using this shap summary plot and
00:09:16.160 then just plug in these
00:09:21.360 values but uh we use three explainer
00:09:24.000 earlier what if you want to explain
00:09:25.880 other models you could use kernel
00:09:27.200 explainer still the same concept but it
00:09:29.640 basically um simulates the model so it's
00:09:32.640 not exactly the same thing but more or
00:09:34.920 less um you're going to get the same
00:09:37.000 explanation and it's a bit
00:09:39.480 slower because it's an
00:09:43.320 approximation so you could use this same
00:09:47.040 visualization you get more or less the
00:09:49.480 same output If You observe carefully you
00:09:51.800 you will see some changes but it's
00:09:54.440 generally the same if you skel explainer
00:09:57.640 and for the ranking it's still
00:10:00.079 the same I mean the ranking of the the
00:10:02.480 feature
00:10:04.480 importances and that's
00:10:13.839 it yeah
