# OpenArmRenderer

This is the minimal UE 5.7 C++ host project. Generated content, the URLab
plugin, build products, and cooked runtimes are intentionally not committed.
Run `scripts/setup_urlab.sh`, import the persistent asset once, and run
`scripts/package_urlab.sh` after the calibration level has been saved as
`/Game/OpenArmRender`.

The runtime controller reads `-OpenArmJob=/absolute/urlab_job.json`, applies the
full OpenCV intrinsic matrix (including off-centre principal point), disables
motion blur/DoF/auto exposure, and configures colocated URLab RGB, depth and
instance streams. URLab's synchronous puppet step owns pose/frame alignment.
