load("@code//config:platform_config.bzl","platform_glob","platform_glob_lib","platform_include_prefix")

vtk_version="9.*"

cc_library(
    name = "vtk",
    hdrs = platform_glob([
      "include/vtk-"+vtk_version+"/**/*.h",
      "include/vtk-"+vtk_version+"/**/*.hxx",
      "include/vtk-"+vtk_version+"/**/*.txx"]),
    srcs = platform_glob_lib(["vtk"+l+"-"+vtk_version for l in [
      "sys",
      "IOCore",
      "CommonCore",
      "CommonMath",
      "CommonDataModel",
      "CommonExecutionModel",
      "FiltersGeneral",
      "FiltersSources",
      "IOPLY",
      "IOGeometry",
      "IOLegacy",
      "IOXML",
      "RenderingContextOpenGL2",
      "RenderingCore",
      "RenderingAnnotation",
      "RenderingFreeType",
      "RenderingGL2PSOpenGL2",
      "RenderingOpenGL2",
      "InteractionStyle",
      "InteractionWidgets",
      "CommonColor",
    ]]),
    #strip_include_prefix = platform_include_prefix("include/vtk-"+vtk_version),
    strip_include_prefix = select({
      "@platforms//os:linux":"include/vtk-9.1",
      "@platforms//os:macos":"local/include/vtk-9.3"
    }),
    visibility = ["//visibility:public"],
)
