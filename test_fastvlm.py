#!/usr/bin/env python3
"""
Test Apple's FastVLM-7B for video understanding
Comparing with current Qwen2.5-Omni-7B setup
"""

import torch
import sys
import os
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
import base64
from io import BytesIO

# Add project to path
sys.path.append(os.getcwd())

from mmagent.utils.video_processing import process_video_clip

def setup_fastvlm():
    """Setup Apple FastVLM-7B model"""
    print("🍎 Loading Apple FastVLM-7B...")
    
    MODEL_ID = "apple/FastVLM-7B"
    IMAGE_TOKEN_INDEX = -200
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        
        print(f"✅ FastVLM loaded successfully!")
        print(f"📱 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        print(f"🔧 Model dtype: {model.dtype}")
        
        return tokenizer, model, IMAGE_TOKEN_INDEX
        
    except Exception as e:
        print(f"❌ Error loading FastVLM: {e}")
        return None, None, None

def process_video_with_fastvlm(video_path, question, tokenizer, model, image_token_index):
    """Process video using FastVLM"""
    print(f"\n🎬 Processing video: {video_path}")
    
    # Extract video data using M3-Agent's processing
    try:
        base64_video, base64_frames, base64_audio = process_video_clip(video_path, fps=5)
        print(f"📹 Extracted {len(base64_frames)} frames")
        
        # Use middle frame as representative
        if not base64_frames:
            print("❌ No frames extracted")
            return None
            
        middle_frame_idx = len(base64_frames) // 2
        frame_b64 = base64_frames[middle_frame_idx]
        
        # Convert base64 to PIL Image
        frame_bytes = base64.b64decode(frame_b64)
        image = Image.open(BytesIO(frame_bytes)).convert("RGB")
        
        print(f"🖼️ Using frame {middle_frame_idx + 1}/{len(base64_frames)} for analysis")
        
        # Build chat messages
        messages = [
            {"role": "user", "content": f"<image>\n{question}"}
        ]
        
        # Render to string and split around <image> token
        rendered = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        pre, post = rendered.split("<image>", 1)
        
        # Tokenize around image
        pre_ids = tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids
        post_ids = tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids
        
        # Insert image token
        img_tok = torch.tensor([[image_token_index]], dtype=pre_ids.dtype)
        input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(model.device)
        attention_mask = torch.ones_like(input_ids, device=model.device)
        
        # Process image
        pixel_values = model.get_vision_tower().image_processor(
            images=image, return_tensors="pt"
        )["pixel_values"]
        pixel_values = pixel_values.to(model.device, dtype=model.dtype)
        
        print("🤖 Generating response...")
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                images=pixel_values,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the generated part
        prompt_length = len(tokenizer.decode(input_ids[0], skip_special_tokens=True))
        answer = response[prompt_length:].strip()
        
        return {
            "frames_processed": len(base64_frames),
            "frame_used": middle_frame_idx + 1,
            "question": question,
            "answer": answer,
            "model": "Apple FastVLM-7B"
        }
        
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_fastvlm.py <video_path> [question]")
        return
    
    video_path = sys.argv[1]
    question = sys.argv[2] if len(sys.argv) > 2 else "What happens in this video? Describe it in detail."
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    # Setup model
    tokenizer, model, image_token_index = setup_fastvlm()
    if not model:
        return
    
    # Process video
    result = process_video_with_fastvlm(video_path, question, tokenizer, model, image_token_index)
    
    if result:
        print("\n" + "="*60)
        print("🎯 FASTVLM VIDEO ANALYSIS RESULTS")
        print("="*60)
        print(f"📹 Frames processed: {result['frames_processed']}")
        print(f"🖼️ Frame analyzed: {result['frame_used']}")
        print(f"❓ Question: {result['question']}")
        print(f"🤖 Model: {result['model']}")
        print(f"💡 Answer: {result['answer']}")
        print("="*60)
    else:
        print("❌ Failed to process video")

if __name__ == "__main__":
    main()

