# Rapidly Exploring Random Trees
## *(for artistic purposes)*
Visualizing Rapidly Exploring Random Trees (RRTs) has some aesthetic appeal. This is the simplest running example I could come up with. To get a visualization run rrt.py, which will create a series of JPG files with RRT plot. Then run
```
ffmpeg -framerate 10 -pattern_type glob -i './iteration_*.jpg' -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4
```
in that folder to produce a small video.
