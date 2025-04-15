cc_library(
    name = "glog",
    hdrs = glob(["include/**/*.hpp"]) + glob(["include/**/*.h"]),
    linkopts = [
        "-L /usr/local/opt/glog/lib -lglog"
    ],
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)
