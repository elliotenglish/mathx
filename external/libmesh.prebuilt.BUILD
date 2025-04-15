cc_library(
    name = "libmesh",
    hdrs = glob(["include/**/*"]),
    srcs = select({
        "@platforms//os:linux":["lib/libmesh_opt.so.0","lib/libtimpi_opt.so.5"],
        "@platforms//os:macos":["lib/libmesh_opt.0.dylib","lib/libtimpi_opt.5.dylib"]
    }),
    strip_include_prefix = "include",
    deps=["@eigen//:eigen","@vtk//:vtk"],
    visibility = ["//visibility:public"]
)
