import { useEffect, useRef, useState } from 'react';

export const useCamera = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    startCamera();
    
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
      
      setStream(mediaStream);
      setError(null);
    } catch (err) {
      setError('Impossible d\'accéder à la caméra');
      console.error('Erreur caméra:', err);
    }
  };

  const captureFrame = () => {
    if (!videoRef.current || !canvasRef.current) {
      console.warn('Video ou canvas non disponible');
      return null;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    // Vérifier que la vidéo est prête
    if (video.readyState !== video.HAVE_ENOUGH_DATA) {
      console.warn('Vidéo pas encore prête:', video.readyState);
      return null;
    }
    
    if (video.videoWidth === 0 || video.videoHeight === 0) {
      console.warn('Dimensions vidéo invalides');
      return null;
    }

    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    try {
      context.drawImage(video, 0, 0);
      const dataURL = canvas.toDataURL('image/jpeg', 0.8);
      
      // Vérifier que le dataURL est valide
      if (!dataURL || dataURL === 'data:,') {
        console.error('DataURL invalide généré');
        return null;
      }
      
      return dataURL;
    } catch (err) {
      console.error('Erreur capture frame:', err);
      return null;
    }
  };

  return {
    videoRef,
    canvasRef,
    captureFrame,
    error
  };
};
