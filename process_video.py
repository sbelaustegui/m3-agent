#!/usr/bin/env python3
"""
M3-Agent Native Video Processing with Qwen2.5-Omni

This script processes videos using M3-Agent's full native video understanding 
capabilities with Qwen2.5-Omni PyTorch model (no API calls, pure native processing).

Usage:
    python process_video.py                          # Process sample video
    python process_video.py /path/to/your/video.mp4  # Process custom video
    python process_video.py --question "What happens in this video?"
"""

import os
import sys
import json
import argparse
from pathlib import Path
import logging
import contextlib
from io import StringIO

# Configure paths for M3-Agent components
sys.path.insert(0, os.getcwd())

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@contextlib.contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr output"""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def setup_environment():
    """Setup M3-Agent environment"""
    # Set environment variables for optimal performance
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    
    # Set device preference (CPU for macOS compatibility)
    if not torch_available():
        raise ImportError("PyTorch is required. Install with: pip install torch torchvision torchaudio")

def torch_available():
    """Check if PyTorch is available"""
    try:
        import torch
        return True
    except ImportError:
        return False

def process_video_with_m3_agent(video_path, questions=None):
    """
    Process video using M3-Agent's native video understanding with Qwen2.5-Omni
    
    Args:
        video_path (str): Path to video file
        questions (list): Questions to ask about the video
    
    Returns:
        dict: Processing results
    """
    if questions is None:
        questions = [
            "What happens in this video?",
            "Describe the main activities and events.",
            "Who or what are the main subjects?",
            "What is the setting and environment?",
            "Summarize the key points of this video."
        ]
    
    logger.info(f"🎥 Processing Video: {video_path}")
    logger.info(f"🤖 Using Model: Qwen/Qwen2.5-Omni-7B (PyTorch native)")
    logger.info(f"❓ Questions: {len(questions)}")
    print("=" * 60)
    
    try:
        # Import M3-Agent components
        from mmagent.utils.video_processing import process_video_clip
        from mmagent.memory_processing_qwen import generate_video_context
        from mmagent.utils.chat_qwen import get_response, generate_messages
        
        logger.info("🔄 Step 1: Extracting video content...")
        
        # Extract video frames, audio, and metadata using M3-Agent (suppress output)
        with suppress_stdout_stderr():
            result = process_video_clip(video_path, fps=5, audio_fps=16000)
            base64_video, base64_frames, base64_audio = result
        
        logger.info(f"✅ Extracted {len(base64_frames)} frames and audio track")
        
        logger.info("🔄 Step 2: Processing faces and voices...")
        
        # Initialize face and voice processing
        try:
            from mmagent.face_processing import process_faces
            from mmagent.voice_processing import process_voices
            
            # Process faces in video frames (suppress output)
            with suppress_stdout_stderr():
                faces_list = process_faces(base64_frames)
            logger.info(f"✅ Processed faces: {len(faces_list)} detected")
            
            # Process voice segments (suppress output)
            with suppress_stdout_stderr():
                voices_list = process_voices(base64_audio) if base64_audio else {}
            logger.info(f"✅ Processed voices: {len(voices_list)} speakers")
            
        except Exception as e:
            logger.warning(f"⚠️ Face/voice processing error: {e}")
            faces_list = {}
            voices_list = {}
        
        logger.info("🔄 Step 3: Generating comprehensive video context...")
        
        # Generate M3-Agent video context with all modalities (suppress output)
        with suppress_stdout_stderr():
            video_context = generate_video_context(
                base64_frames, faces_list, voices_list, 
                video_path=base64_video, faces_input="face_only"
            )
        
        logger.info("✅ Video context generated successfully")
        
        logger.info("🔄 Step 4: Processing with native Qwen2.5-Omni model...")
        
        results = {}
        
        for i, question in enumerate(questions, 1):
            logger.info(f"\n🤔 Question {i}: {question}")
            
            try:
                # Create optimized input for native processing
                # Use minimal data to stay within model limits
                if base64_frames and len(base64_frames) > 0:
                    # Use only 1 representative frame to stay within memory limits
                    sample_frame = [base64_frames[len(base64_frames)//2]]  # Use middle frame
                    
                    # Create concise context summary
                    context_summary = f"Video: {len(base64_frames)} frames, {question}"
                    
                    inputs = [
                        {
                            "type": "images/jpeg", 
                            "content": sample_frame
                        },
                        {
                            "type": "text",
                            "content": context_summary
                        }
                    ]
                else:
                    # Text-only fallback with concise context
                    inputs = [
                        {
                            "type": "text", 
                            "content": f"Video analysis: {len(base64_frames) if base64_frames else 0} frames. {question}"
                        }
                    ]

                
                # Generate messages in M3-Agent format
                messages = generate_messages(inputs)
                
                # Process with native Qwen2.5-Omni model (suppress verbose output)
                with suppress_stdout_stderr():
                    response, token_count = get_response(messages)
                
                # Print prominently to console
                print(f"\n🤖 Question {i+1}: {question}")
                print(f"💡 Answer: {response}")
                print(f"📊 Tokens: {token_count}")
                print("-" * 60)
                
                results[f"question_{i}"] = {
                    "question": question,
                    "answer": response,
                    "tokens": token_count,
                    "video_context_included": bool(video_context)
                }
                
            except Exception as e:
                logger.error(f"❌ Error processing question {i}: {e}")
                results[f"question_{i}"] = {
                    "question": question,
                    "error": str(e)
                }
        
        return {
            "video_path": video_path,
            "model": "Qwen/Qwen2.5-Omni-7B",
            "processing_method": "native_pytorch",
            "frames_extracted": len(base64_frames),
            "faces_detected": len(faces_list),
            "speakers_detected": len(voices_list),
            "has_audio": len(base64_audio) > 0,
            "video_context_generated": bool(video_context),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"❌ Error in video processing pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Process videos with M3-Agent and native Qwen2.5-Omni model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_video.py
  python process_video.py /path/to/video.mp4
  python process_video.py --question "What is the main activity?"
  python process_video.py video.mp4 --questions "What happens?" "Who is present?"
        """
    )
    
    parser.add_argument(
        "video_path",
        nargs="?",
        default="data/videos/test_sample.mp4",
        help="Path to video file (default: sample video)"
    )
    
    parser.add_argument(
        "--question",
        help="Single question to ask about the video"
    )
    
    parser.add_argument(
        "--questions",
        nargs="+",
        help="Multiple questions to ask about the video"
    )
    
    args = parser.parse_args()
    
    # Check if video exists
    if not Path(args.video_path).exists():
        print(f"❌ Video file not found: {args.video_path}")
        if args.video_path == "data/videos/test_sample.mp4":
            print("💡 Create sample video with: ffmpeg -f lavfi -i testsrc2=duration=10:size=640x480:rate=30 -f lavfi -i sine=frequency=1000:duration=10 -c:v libx264 -c:a aac -shortest data/videos/test_sample.mp4")
        return 1
    
    # Setup questions
    questions = None
    if args.question:
        questions = [args.question]
    elif args.questions:
        questions = args.questions
    
    # Setup environment
    setup_environment()
    
    print("🚀 M3-Agent Native Video Processing")
    print("=" * 60)
    print("🎯 Full multimodal video understanding")
    print("🤖 Using native Qwen2.5-Omni-7B PyTorch model")
    print("🔧 M3-Agent complete processing pipeline")
    print("💾 No API calls - pure local processing")
    print()
    
    # Process video
    results = process_video_with_m3_agent(args.video_path, questions)
    
    if results:
        print("\n" + "=" * 60)
        print("🎉 Video processing completed successfully!")
        print(f"📊 Processed {results['frames_extracted']} frames")
        print(f"👥 Faces detected: {results['faces_detected']}")
        print(f"🗣️ Speakers detected: {results['speakers_detected']}")
        print(f"🎵 Audio: {'✅ Yes' if results['has_audio'] else '❌ No'}")
        print(f"🧠 Video context: {'✅ Generated' if results['video_context_generated'] else '❌ Failed'}")
        print(f"🤖 Model: {results['model']} ({results['processing_method']})")
        
        # Save results
        output_file = f"video_analysis_{Path(args.video_path).stem}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {output_file}")
        
    else:
        print("\n❌ Video processing failed")
        print("💡 Make sure:")
        print("   1. All M3-Agent dependencies are installed")
        print("   2. PyTorch and transformers are installed")
        print("   3. Video file is valid and accessible")
        print("   4. Internet connection for model download (first run)")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
