#for f in /usr/local/lib/libopencv*.4.5.5.dylib; do echo $f; nm $f | grep farneback; done

load("@code//config:platform_config.bzl","platform_glob","platform_glob_lib","platform_include_prefix")

cc_library(
    name = "opencv",
    hdrs = platform_glob(["include/opencv*/**/*.hpp","include/opencv*/**/*.h"]),
    srcs = platform_glob_lib(["opencv_"+ l for l in [
        "calib3d",
        "imgcodecs",
        "imgproc",
        "videoio",
        "viz",
        "highgui",
        "flann",
        #"xfeatures2d",
        "video",
        "features2d",
        "core",
    ]]),
    strip_include_prefix = platform_include_prefix("opencv4/"),
    visibility = ["//visibility:public"],
)
