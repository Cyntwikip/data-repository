# tactiq.io free youtube transcript
# Model Interpretability - Introduction
# https://www.youtube.com/watch/bx5_QNJmpJs

00:00:02.560 hello everyone so in this session we'll
00:00:05.480 be covering model
00:00:08.639 interpretability so let's
00:00:12.320 start but first of all I'm going to show
00:00:14.719 you these uh different machine learning
00:00:17.240 use cases so on the left hand side
00:00:19.840 you're seeing e-commerce product
00:00:21.519 recommendations image filters and
00:00:23.279 enhancements spam detection social media
00:00:26.160 feed ranking real-time object detection
00:00:29.000 and on the right hand side
00:00:30.640 for the use cases to you're seeing
00:00:32.000 credit scoring Health Diagnostics
00:00:34.200 medical treatment recommendations fraud
00:00:35.840 detection and H end
00:00:38.239 recruitment so off the top of your head
00:00:40.320 what are you seeing what are the
00:00:41.800 differences of the two use cases
00:00:46.640 here I would say the left hand side
00:00:51.480 would be um would require more
00:00:56.760 um performance or or accuracy
00:01:00.800 in uh their outputs whereas these cases
00:01:02.800 2 we need more explainability of course
00:01:05.560 um that's the intent of this uh session
00:01:07.400 because it's all about
00:01:10.400 interpretability so in some tasks
00:01:12.759 getting the prediction the what isn't
00:01:15.159 enough the model also needs to explain
00:01:17.799 how it arrived at that prediction the
00:01:20.479 why so let's go back to the examples
00:01:22.840 that I've given earlier so on the left
00:01:25.119 hand side it's about the what and the
00:01:27.759 right hand side it's about the why so
00:01:29.680 why is it that uh we have more emphasis
00:01:33.280 on the model performance on the left
00:01:35.960 hand side it is because these are well
00:01:39.680 established use cases um and even though
00:01:42.280 we cannot or even if we don't exactly
00:01:45.640 give out the rational for the decision
00:01:48.640 to the intended audience it still works
00:01:51.439 it's okay like when you use your filters
00:01:54.280 in your phone right uh to enhance your
00:01:56.240 images do you really have to understand
00:01:58.600 how it did that you don't write same
00:02:01.680 goes on the or the opposite applies on
00:02:04.399 the other um use cases here on the right
00:02:07.920 hand side so there's more emphasis on
00:02:10.759 interpretability like say for example
00:02:12.640 credit score if you are rejected or
00:02:15.879 approved um when you apply for a loan or
00:02:18.920 credit card you want to understand why
00:02:21.720 and it's also the responsibility of
00:02:23.280 those institutions to explain to you how
00:02:26.000 they arrive to that conclusion and
00:02:28.160 especially in the context of Medicine
00:02:30.560 in health right uh you don't just
00:02:33.640 blindly accept what the machine learning
00:02:35.599 model uh will tell you you also want to
00:02:38.480 know why it led to that
00:02:41.360 recommendation so that's the reason why
00:02:43.360 we need model interpretability again not
00:02:45.400 all these cases need
00:02:47.680 interpretability um but a lot of Al also
00:02:51.319 these cases that we have in the industry
00:02:52.840 would need
00:02:57.080 interpretability so how do we
00:03:00.640 enable those when we create our machine
00:03:02.480 learning models right so we have two
00:03:05.959 main approaches first is using
00:03:07.879 interpretable models and then the other
00:03:10.120 one is model antic methods so for the
00:03:12.440 interpretable models we use models that
00:03:14.360 are inherently interpretable such as the
00:03:17.040 linear and free based models they're
00:03:19.159 very straightforward whereas the model
00:03:21.280 agnostic methods we use these models um
00:03:25.200 because of their flexibility we use them
00:03:28.239 for non-interpretable
00:03:30.200 models and it is because they can apply
00:03:32.560 to can any model even those as complex
00:03:35.799 as deep neural networks
