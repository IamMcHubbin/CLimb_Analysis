# Sample footage

Drop climbing clips here to benchmark and tune against real video. Committed
deliberately, so `.gitignore` has an exception for this directory.

Add them straight off the phone — **do not re-encode to shrink a file**. A
re-export strips the variable frame timing and the rotation flag, which are
exactly what the ingest path exists to handle, so a converted clip tests
nothing. To cut the size, trim instead; this copies the stream untouched:

```bash
ffmpeg -ss 0 -t 30 -i IMG_1234.MOV -c copy fixtures/climb.MOV
```

Useful clips are 20-60 seconds and include the awkward cases: another person
walking through frame, a belayer in shot, the climber small and high on the
wall. Those are where tracking is expected to break.
