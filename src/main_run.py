"""
Main run script for testing Petrolimex AI Backend
Test ThreadManager với camera và AI models
"""
import time
import logging
from uuid import UUID
from src.core.thread_manager import thread_manager
from src.core.performance_monitor import performance_monitor
from src.database import SessionLocal
from src.models.camera import Camera
from src.models.location import Location
from src.models.ai_model import AiModel
from src.models.camera_model import CameraModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def system_info():
    """Hiển thị thông tin hệ thống"""
    import torch
    
    logger.info("=" * 60)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 60)
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
        gpu_count = torch.cuda.device_count()
        logger.info(f"Number of CUDA Devices: {gpu_count}")
        for i in range(gpu_count):
            logger.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info("No CUDA-capable device found. Running on CPU.")
    logger.info("=" * 60)


def show_database_status():
    """Hiển thị trạng thái database"""
    db = SessionLocal()
    try:
        cameras = db.query(Camera).all()
        ai_models = db.query(AiModel).all()
        camera_models = db.query(CameraModel).filter(CameraModel.is_enabled == True).all()
        
        logger.info("=" * 60)
        logger.info("DATABASE STATUS")
        logger.info("=" * 60)
        logger.info(f"Total Cameras: {len(cameras)}")
        for cam in cameras:
            logger.info(f"  - {cam.name} (ID: {cam.id}, Status: {'Active' if cam.status else 'Inactive'})")
        
        logger.info(f"\nTotal AI Models: {len(ai_models)}")
        for model in ai_models:
            logger.info(f"  - {model.name} v{model.version} (Active: {model.is_active})")
        
        logger.info(f"\nEnabled Camera-Model Relationships: {len(camera_models)}")
        for cm in camera_models:
            logger.info(f"  - Camera {cm.camera_id} <-> Model {cm.model_id}")
        logger.info("=" * 60)
        
    finally:
        db.close()


def test_single_camera(camera_id: str, duration: int = 60):
    """
    Test chạy một camera trong khoảng thời gian nhất định
    
    Args:
        camera_id: UUID của camera dạng string
        duration: Thời gian chạy test (giây)
    """
    try:
        cam_uuid = UUID(camera_id)
        logger.info(f"\n{'=' * 60}")
        logger.info(f"TESTING SINGLE CAMERA: {camera_id}")
        logger.info(f"Duration: {duration} seconds")
        logger.info(f"{'=' * 60}\n")
        
        # Start camera
        logger.info("Starting camera...")
        success = thread_manager.start_camera(cam_uuid)
        
        if not success:
            logger.error("Failed to start camera!")
            return
        
        logger.info("Camera started successfully!")
        logger.info(f"Running for {duration} seconds...")
        logger.info(f"Status: {thread_manager.get_status()}\n")
        
        # Run for specified duration
        time.sleep(duration)
        
        # Stop camera
        logger.info("\nStopping camera...")
        thread_manager.stop_camera(cam_uuid)
        logger.info("Camera stopped!")
        
    except ValueError:
        logger.error(f"Invalid camera_id format: {camera_id}")
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        try:
            thread_manager.stop_camera(cam_uuid)
        except KeyboardInterrupt:
            logger.warning("Force stopping... Please wait for threads to terminate.")
    except Exception as e:
        logger.error(f"Test error: {e}")


def test_all_active_cameras(duration: int = 60):
    """
    Test chạy tất cả cameras có status=True
    
    Args:
        duration: Thời gian chạy test (giây)
    """
    try:
        logger.info(f"\n{'=' * 60}")
        logger.info("TESTING ALL ACTIVE CAMERAS")
        logger.info(f"Duration: {duration} seconds")
        logger.info(f"{'=' * 60}\n")
        
        # Load all active cameras
        logger.info("Loading all active cameras...")
        thread_manager.load_all_active_cameras()

        # Start performance monitor
        logger.info("Starting performance monitor...")
        performance_monitor.start()

        status = thread_manager.get_status()
        logger.info(f"Loaded cameras: {status}")

        if status['total_cameras'] == 0:
            logger.warning("No active cameras found!")
            performance_monitor.stop()
            return

        logger.info(f"\nRunning {status['total_cameras']} camera(s) for {duration} seconds...")
        logger.info("Performance metrics will be saved every 10 minutes")
        logger.info("Press Ctrl+C to stop early\n")

        # Run for specified duration
        time.sleep(duration)
        
        # Stop all
        logger.info("\nStopping all cameras...")
        thread_manager.stop_all()
        logger.info("All cameras stopped!")

        logger.info("Stopping performance monitor...")
        performance_monitor.stop()
        logger.info("Performance monitor stopped!")

    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        try:
            thread_manager.stop_all()
            performance_monitor.stop()
        except KeyboardInterrupt:
            logger.warning("Force stopping... Please wait for threads to terminate.")
    except Exception as e:
        logger.error(f"Test error: {e}")
        try:
            thread_manager.stop_all()
            performance_monitor.stop()
        except KeyboardInterrupt:
            logger.warning("Interrupted during cleanup")


def test_model_reload(camera_id: str):
    """
    Test reload models cho camera
    
    Args:
        camera_id: UUID của camera dạng string
    """
    try:
        cam_uuid = UUID(camera_id)
        logger.info(f"\n{'=' * 60}")
        logger.info(f"TESTING MODEL RELOAD: {camera_id}")
        logger.info(f"{'=' * 60}\n")
        
        # Start camera
        logger.info("Starting camera...")
        thread_manager.start_camera(cam_uuid)
        time.sleep(5)
        
        # Reload models
        logger.info("Reloading models...")
        success = thread_manager.reload_models(cam_uuid)
        
        if success:
            logger.info("Models reloaded successfully!")
        else:
            logger.error("Failed to reload models!")
        
        time.sleep(5)
        
        # Stop camera
        logger.info("Stopping camera...")
        thread_manager.stop_camera(cam_uuid)
        logger.info("Test completed!")
        
    except ValueError:
        logger.error(f"Invalid camera_id format: {camera_id}")
    except Exception as e:
        logger.error(f"Test error: {e}")


def interactive_menu():
    """Menu tương tác để test"""
    while True:
        print("\n" + "=" * 60)
        print("PETROLIMEX AI BACKEND - TEST MENU")
        print("=" * 60)
        print("1. Show system info")
        print("2. Show database status")
        print("3. Test single camera")
        print("4. Test all active cameras")
        print("5. Test model reload")
        print("6. Show current status")
        print("0. Exit")
        print("=" * 60)
        
        choice = input("\nEnter your choice: ").strip()
        
        if choice == "1":
            system_info()
        elif choice == "2":
            show_database_status()
        elif choice == "3":
            camera_id = input("Enter camera ID (UUID): ").strip()
            duration = input("Enter duration in seconds (default 60): ").strip()
            duration = int(duration) if duration else 60
            test_single_camera(camera_id, duration)
        elif choice == "4":
            duration = input("Enter duration in seconds (default 60): ").strip()
            duration = int(duration) if duration else 60
            test_all_active_cameras(duration)
        elif choice == "5":
            camera_id = input("Enter camera ID (UUID): ").strip()
            test_model_reload(camera_id)
        elif choice == "6":
            status = thread_manager.get_status()
            logger.info(f"Current status: {status}")
        elif choice == "0":
            logger.info("Exiting...")
            try:
                thread_manager.stop_all()
            except KeyboardInterrupt:
                logger.warning("Interrupted during shutdown. Force exiting...")
            break
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    import sys
    
    # System info
    system_info()
    
    # Nếu có arguments, chạy theo mode
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "info":
            show_database_status()
        elif mode == "test-all":
            duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
            test_all_active_cameras(duration)
        elif mode == "test-camera":
            if len(sys.argv) < 3:
                logger.error("Usage: python -m src.main_run test-camera <camera_id> [duration]")
                sys.exit(1)
            camera_id = sys.argv[2]
            duration = int(sys.argv[3]) if len(sys.argv) > 3 else 60
            test_single_camera(camera_id, duration)
        elif mode == "interactive":
            interactive_menu()
        else:
            logger.error(f"Unknown mode: {mode}")
            logger.info("Available modes: info, test-all, test-camera, interactive")
    else:
        # Default: interactive menu
        interactive_menu()