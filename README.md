Code adapted from https://github.com/leafqycc/rknn-cpp-Multithreading/ , licensed under Apache 2.0. JNI written around this code and licensed under GPL V3.

Instructions for the model conversion process can be found [here](https://docs.photonvision.org/en/latest/docs/objectDetection/about-object-detection.html#training-custom-models).

In order to build this code, it's necessary to run ``gradle`` using the ``publishToMavenLocal`` argument.  Then, in the ``build.gradle`` file of the other repository, update the dependency to reference the locally published JAR.
