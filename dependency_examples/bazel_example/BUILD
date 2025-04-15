load("//:utils.bzl","generate_value")

# some_value=select({
#     "@platforms//os:macos":"this is macros",
#     "@platforms//os:linux":"this is linux"})

cc_binary(
  name="test_bin",
  srcs=["test.cpp"],
  copts=select({
   "@platforms//os:macos":["'-DSWITCHED_MACRO=\"this is macos\"'"],
   "@platforms//os:linux":["'-DSWITCHED_MACRO=\"this is linux\"'"]
  }),
  # copts=["'-DSWITCHED_MACRO=\""+some_value+"\"'"],
)
