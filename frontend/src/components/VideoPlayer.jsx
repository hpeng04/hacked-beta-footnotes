import { forwardRef } from "react";

const VideoPlayer = forwardRef(function VideoPlayer({ src }, ref) {
  return <video ref={ref} src={src || undefined} controls className="video" />;
});

export default VideoPlayer;
