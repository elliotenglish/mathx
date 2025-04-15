cc_library(
    name = "gflags",
    hdrs = glob(["include/**/*.hpp"]) + glob(["include/**/*.h"]),
    linkopts = [
        "-L /usr/local/opt/gflags/lib -lgflags"
    ],
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)
