"""
Performance Monitor - Background thread to collect system metrics
Runs every 10 minutes and saves to database
"""
import time
import logging
import psutil
from threading import Thread, Event
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

from src.database import SessionLocal
from src.models.performance import Performance
from src.models.event import Event as EventModel
from src.core.thread_manager import thread_manager

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor system performance and save metrics"""

    def __init__(self, interval_seconds: int = 600):  # Default: 10 minutes
        """
        Initialize performance monitor

        Args:
            interval_seconds: Interval between measurements (default: 600s = 10min)
        """
        self.interval_seconds = interval_seconds
        self.stop_event = Event()
        self.thread = None
        self.last_frame_counts = {}  # {camera_id: frame_count}
        self.last_measurement_time = None

    def start(self):
        """Start monitoring thread"""
        if self.thread is not None and self.thread.is_alive():
            logger.warning("PerformanceMonitor already running")
            return

        self.stop_event.clear()
        self.thread = Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info(f"PerformanceMonitor started (interval: {self.interval_seconds}s)")

    def stop(self):
        """Stop monitoring thread"""
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
        logger.info("PerformanceMonitor stopped")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self.stop_event.is_set():
            try:
                metrics = self.collect_metrics()
                self.save_metrics(metrics)
                logger.info(f"Performance metrics saved: CPU={metrics['cpu_percent']:.1f}% "
                           f"MEM={metrics['memory_percent']:.1f}% "
                           f"FPS={metrics['total_fps']:.1f}")
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}", exc_info=True)

            # Sleep in small intervals to allow quick shutdown
            for _ in range(self.interval_seconds):
                if self.stop_event.is_set():
                    break
                time.sleep(1)

    def collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        current_time = datetime.now()

        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / (1024 * 1024)

        # Camera metrics
        camera_status = thread_manager.get_status()
        running_cameras = camera_status['total_cameras']
        camera_ids = camera_status['cameras']

        # Calculate FPS for each camera
        camera_details = {}
        total_fps = 0.0
        total_models = 0

        for camera_id in camera_ids:
            camera_manager = thread_manager.cameras.get(camera_id)
            if camera_manager:
                # Estimate FPS from queue activity
                queue_size = camera_manager.frame_queue.qsize()
                num_models = len(camera_manager.processor_thread.processor.models)

                # Simple FPS estimation (can be improved with actual frame counting)
                fps = self._estimate_camera_fps(camera_id, queue_size)

                camera_details[str(camera_id)] = {
                    'fps': fps,
                    'models': num_models,
                    'queue_size': queue_size
                }

                total_fps += fps
                total_models += num_models

        avg_fps = total_fps / running_cameras if running_cameras > 0 else 0.0

        # Event metrics
        db = SessionLocal()
        try:
            total_events = db.query(EventModel).count()

            # Events in last hour
            one_hour_ago = current_time - timedelta(hours=1)
            events_last_hour = db.query(EventModel).filter(
                EventModel.time >= one_hour_ago
            ).count()
        finally:
            db.close()

        # Get total cameras from database
        from src.models.camera import Camera
        db = SessionLocal()
        try:
            total_cameras = db.query(Camera).count()
        finally:
            db.close()

        return {
            'timestamp': current_time,
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'memory_used_mb': memory_used_mb,
            'total_cameras': total_cameras,
            'running_cameras': running_cameras,
            'total_models': total_models,
            'total_fps': total_fps,
            'avg_fps_per_camera': avg_fps,
            'total_events': total_events,
            'events_last_hour': events_last_hour,
            'camera_details': camera_details
        }

    def _estimate_camera_fps(self, camera_id, queue_size: int) -> float:
        """
        Estimate FPS for a camera
        Simple estimation based on queue changes (can be improved)
        """
        # Default FPS estimation
        # If queue is not empty, camera is processing frames
        if queue_size > 0:
            return 15.0  # Approximate FPS (adjust based on actual performance)
        return 0.0

    def save_metrics(self, metrics: Dict[str, Any]):
        """Save metrics to database"""
        db = SessionLocal()
        try:
            performance = Performance(
                timestamp=metrics['timestamp'],
                cpu_percent=metrics['cpu_percent'],
                memory_percent=metrics['memory_percent'],
                memory_used_mb=metrics['memory_used_mb'],
                total_cameras=metrics['total_cameras'],
                running_cameras=metrics['running_cameras'],
                total_models=metrics['total_models'],
                total_fps=metrics['total_fps'],
                avg_fps_per_camera=metrics['avg_fps_per_camera'],
                total_events=metrics['total_events'],
                events_last_hour=metrics['events_last_hour'],
                camera_details=metrics['camera_details']
            )
            db.add(performance)
            db.commit()
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
            db.rollback()
        finally:
            db.close()

    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get latest metrics (for real-time display)"""
        try:
            return self.collect_metrics()
        except Exception as e:
            logger.error(f"Error getting latest metrics: {e}")
            return {}


# Global instance
performance_monitor = PerformanceMonitor(interval_seconds=600)  # 10 minutes
