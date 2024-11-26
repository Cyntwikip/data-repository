# tactiq.io free youtube transcript
# Model Interpretability - Interpretable Models Coding
# https://www.youtube.com/watch/KxpbawRIagM

00:00:04.480 okay so let's start coding interpretable
00:00:09.679 models as usual I'll be importing some
00:00:12.160 of the libraries and in this notebook
00:00:15.280 I'll be using the Titanic data
00:00:18.400 set I've also done um some of the
00:00:21.519 pre-processing here such as the train
00:00:23.599 test split um encoding and utation of um
00:00:28.519 null values
00:00:33.800 taking into consideration the data
00:00:35.480 leakage as well so if I've done this
00:00:37.640 correctly
00:00:38.800 for now let's discuss interpretable
00:00:41.640 models and uh the simplest approach is
00:00:45.239 use models that are inherently
00:00:46.920 interpretable and common examples of
00:00:49.039 that would be linear regression listic
00:00:50.800 regression decision trees so for linear
00:00:54.039 model is very good because it's very
00:00:57.120 straightforward and these models have
00:00:58.920 been used by statisticians and data
00:01:01.559 scientists for
00:01:07.159 decades in the relationship essentially
00:01:10.320 y mx plus
00:01:13.400 P but do take note of uh the assumptions
00:01:17.159 for linear aggression such as linearity
00:01:19.640 H muscularity normality of erors and
00:01:22.280 multicolinearity
00:01:23.759 interpretation is very straightforward
00:01:26.600 because every one unit increase and if
00:01:29.479 each X subi the predicted outcome
00:01:31.880 changes by Beta
00:01:34.240 subi for this notebook though we will be
00:01:36.680 using logistic regression since that was
00:01:39.040 the classification version of the
00:01:40.600 problem but the same concept still
00:01:43.840 applies so I've done some um scaling
00:01:46.680 here first uh before I trained
00:01:51.240 it and uh I printed out a score it's 81%
00:01:55.479 but I also printed out the
00:01:58.079 coefficients I did the scaling so that
00:02:01.320 um we uh have the same unit because um
00:02:07.159 originally the data set would have
00:02:09.800 varying um scales could be thousands
00:02:12.720 millions some would be decimal so by
00:02:15.480 standardizing it using the standard
00:02:17.800 scaler um the
00:02:19.760 contribution will more or less uh or the
00:02:22.560 weights or the the the value the scale
00:02:25.120 of the values would be um in the same
00:02:28.200 range now um I've printed out the
00:02:30.840 coefficients you can see the impact that
00:02:33.640 it has to the
00:02:36.760 output and that's how we check the
00:02:39.800 feature importance but I've also um done
00:02:44.159 some data manipulation here so um we
00:02:48.239 could clearly see the
00:02:51.400 ranking I have created a visualization
00:02:53.800 here same bar graph you can see that the
00:02:56.640 top predictor is the sex by class age
00:03:01.040 and
00:03:02.200 others so generally having high sex male
00:03:08.120 value which essentially means that the
00:03:10.080 person is male um negatively impacts the
00:03:15.120 survivable of uh the specific
00:03:20.200 person so generally that's how it goes
00:03:23.120 with all other features except Fair
00:03:26.080 here basically if you have a higher fair
00:03:28.879 then that's better
00:03:30.280 for the rest it's bad if you have higher
00:03:35.760 value yeah as you can see if the sexis
00:03:38.319 ma the survival rate decreases
00:03:40.519 significantly another module that we
00:03:43.040 could
00:03:43.760 use um for interpretability purposes as
00:03:46.760 tree based model like decision
00:03:50.760 tree a decision tree splits a database
00:03:53.439 on certain feature thresholds forming
00:03:55.319 subsets until reaching terminal nodes or
00:03:57.519 the Le nodes
00:04:00.480 so we have trained the model
00:04:01.760 interpretation is very straightforward
00:04:04.040 we start at the root to
00:04:06.319 here and then um based on the value that
00:04:10.439 it has on that specific feature will go
00:04:12.640 to the left or the right uh branch and
00:04:16.120 you can see the the class that it will
00:04:19.279 have and then just keep on following the
00:04:22.720 if else rules up until you get to the
00:04:26.199 leaf node and that will be your final
00:04:28.120 prediction
00:04:30.479 and you could also explain this again to
00:04:33.400 um non-technical stakeholders or
00:04:36.479 audience so how do we check the feature
00:04:40.720 importance um it is determined by how
00:04:43.000 much it reduces the impurities the
00:04:45.039 impurity across all splits that use that
00:04:47.440 feature and the importances are
00:04:49.560 normalized to some to
00:04:51.800 100% and you could use the built and
00:04:54.720 feature importances attribute of the
00:04:57.440 model here n SK learn so you see that
00:05:01.520 still the top predictor is sex
00:05:05.280 male and that's the output which I
00:05:09.120 visualized
00:05:11.720 here and that's it for
00:05:14.680 this notebook session
