import json, os, time
from os import path
from openai import OpenAI
import mimetypes, base64
import numpy as np
import cv2

class LLMAPI:

    def __init__(self, model_name, api_key, base_url) -> None:
        """
        Initializes the LLM API client and conversation history.
        """
        print(f"Initializing: using the model {model_name}")
        self.model_name = model_name

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0, # We handle retries manually
            timeout=100
        )
        self.error_occur = 0
        self.history = [] # Initialize conversation history
        print("Client has been created.")

    def chat(self, question: str, 
             images: list = None, 
             videos: list = None, 
             video_fps: float = 0.5,
             video_resolution: int = 224,
             **kwargs) -> str:
        """
        Unified interface for chat, supporting text, images, and videos.
        Automatically manages the conversation history.

        Args:
            question (str): User question.
            images (list, optional): List of image paths or NumPy arrays. Defaults to None.
            videos (list, optional): List of video paths. Defaults to None.
            video_fps (float, optional): Frames per second to extract from videos. Defaults to 0.5.
            video_resolution (int, optional): The max resolution for extracted frames. Defaults to 224.
            **kwargs: Additional arguments for the API call, such as temperature, max_tokens.

        Returns:
            str: The model's response.
        """
        images = images or []
        videos = videos or []
        
        # Prepare the content for the user's message
        user_content = []
        
        # Process images
        for img_source in images:
            encoded_image = self._encode_image_to_base64(img_source)
            if encoded_image:
                user_content.append({
                    "type": "image",
                    "image": encoded_image
                })
        
        # Process videos
        for video_path in videos:
            try:
                frames = self._extract_frames(video_path, fps=video_fps, max_resolution=video_resolution)
                print(f"Extracted {len(frames)} frames from {os.path.basename(video_path)}.")
                for frame in frames:
                    encoded_frame = self._encode_image_to_base64(frame)
                    if encoded_frame:
                        user_content.append({
                            "type": "image_url",
                            "image_url": {"url": encoded_frame}
                        })
            except (ValueError, FileNotFoundError) as e:
                print(f"Warning: Could not process video '{video_path}'. {e}. Skipping.")

        # Add the text question at the end
        user_content.append({"type": "text", "text": question})
        
        # Create and store the user message
        user_message = {"role": "user", "content": user_content}
        self.history.append(user_message)
        
        # Generate a response from the model
        assistant_response = self._generate(**kwargs)
        
        # Create and store the assistant's response
        assistant_message = {"role": "assistant", "content": [{"type": "text", "text": assistant_response}]}
        self.history.append(assistant_message)
        
        return assistant_response

    def _generate(self, **kwargs) -> str:
        """
        Private method to make the API call and handle errors/retries.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.history,
                stream=False,
                **kwargs # Pass any extra parameters like temperature, max_tokens
            )
            answer = response.choices[0].message.content
            self.error_occur = 0  # Reset error count on success
            return answer
        except Exception as e:
            self.error_occur += 1
            print(f"Error during API call: {e}")
            self._process_error()

            return self._generate(**kwargs)

    def _process_error(self, sleep_time=3, max_errors=20):
        """
        Handles errors by sleeping and raising an exception if too many errors occur.
        """
        if self.error_occur > max_errors:
            raise RuntimeError(f"Too many errors occurred: {self.error_occur}. Aborting.")
        print(f"Retrying in {sleep_time} seconds... (Attempt {self.error_occur})")
        time.sleep(sleep_time)

    def _encode_image_to_base64(self, image_source):
        """
        Encodes an image (from file path or NumPy array) to a Base64 Data URI.
        """
        try:
            if isinstance(image_source, np.ndarray):
                # Handle NumPy array input
                is_success, buffer = cv2.imencode(".jpg", image_source)
                if not is_success:
                    raise ValueError("Failed to encode NumPy array to JPEG")
                binary_data = buffer.tobytes()
                mime_type = "image/jpeg"
            else:
                # Handle file path input
                if not os.path.exists(image_source):
                    raise FileNotFoundError(f"File not found at '{image_source}'")
                mime_type, _ = mimetypes.guess_type(image_source)
                if not mime_type or not mime_type.startswith("image"):
                    mime_type = "image/jpeg"
                with open(image_source, "rb") as image_file:
                    binary_data = image_file.read()

            base64_encoded_data = base64.b64encode(binary_data).decode("utf-8")
            return f"data:{mime_type};base64,{base64_encoded_data}"
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None

    def _extract_frames(self, video_path: str, fps: float, max_resolution: int, max_frames: int = 96) -> list:
        """
        Extracts frames from a video file at a specified rate.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(video_fps / fps)) # Calculate how many frames to skip
        
        extracted_frames = []
        frame_idx = 0
        while cap.isOpened() and len(extracted_frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                resized_frame = self._resize_frame(frame, max_resolution)
                extracted_frames.append(resized_frame)
            
            frame_idx += 1
            
        cap.release()
        return extracted_frames

    @staticmethod
    def _resize_frame(frame: np.ndarray, resolution: int) -> np.ndarray:
        """
        Resizes a single frame while maintaining aspect ratio.
        """
        original_height, original_width = frame.shape[:2]
        if max(original_height, original_width) <= resolution:
            return frame
        
        scaling_factor = resolution / max(original_height, original_width)
        
        new_width = int(original_width * scaling_factor)
        new_height = int(original_height * scaling_factor)
        
        resized_image = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized_image

    def clear_history(self):
        """
        Clears the conversation history.
        """
        self.history = []
        print("Conversation history has been cleared.")


if __name__ == "__main__":
    # Initialize the client
    llm = LLMAPI(
        model_name="Qwen2.5-VL-7B",
        base_url="http://172.31.233.64:2559/local",
        api_key="root",
    )
    # llm = LLMAPI(model_name="Lingshu-32B",
    #         api_key="root",
    #         base_url="http://172.31.58.9:5521/v1")

    # # --- Example 1: Simple text question ---
    # print("\n--- Text-only Example ---")
    # answer_text = llm.chat("What's your name?")
    # print("Model Response:", answer_text)

    # # --- Example 2: Ask a follow-up question (demonstrates history) ---
    # print("\n--- Follow-up Example ---")
    # follow_up_answer = llm.chat("What just I asked you?")
    # print("Model Response:", follow_up_answer)

    # Clear history for a new conversation
    llm.clear_history()

    # --- Example 3: Image question ---
    print("\n--- Image Example ---")
    # Make sure this image path is correct on your system
    image_path = "/data/yangwennuo/code/MNL/genWhat/OIP-C.jpg" 
    if os.path.exists(image_path):
        answer_image = llm.chat("请详细描述这张图片。", images=[image_path])
        print("Model Response:", answer_image)
    else:
        print(f"Image not found at {image_path}, skipping image example.")
    
    answer_image = llm.chat("这张图片有什么特别的地方吗？")
    print("Model Response:", answer_image)

    # --- Example 4: Video question (optional) ---
    # print("\n--- Video Example ---")
    # # Make sure this video path is correct on your system
    # video_path = "./case_0422.mp4"
    # if os.path.exists(video_path):
    #     answer_video = llm.chat("请总结一下这个视频的内容。", videos=[video_path], video_fps=0.3) # Extract 1 frame every ~3 seconds
    #     print("Model Response:", answer_video)
    # else:
    #     print(f"Video not found at {video_path}, skipping video example.")