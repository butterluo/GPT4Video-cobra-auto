import logging

lg = logging.getLogger("CBRWEB")
lg.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s ][%(name)s][ %(levelname)s )( %(message)s")
ch.setFormatter(formatter)
lg.addHandler(ch)

# logging.basicConfig(level=lg.INFO,
#                format="%(asctime)s ][%(name)s] %(levelname)s )( %(message)s")

import gradio as gr
import os
import json
import time
import threading

import gradio as gr
import cv2
import os
import base64
import requests
from moviepy.editor import VideoFileClip
import logging
import json
import sys
import threading
logging.getLogger('moviepy').setLevel(logging.ERROR)
import time
from functools import wraps
from dotenv import load_dotenv


load_dotenv()
vision_endpoint=os.environ["VISION_ENDPOINT"]
vision_api_type=os.environ["VISION_API_TYPE"]
vision_deployment=os.environ["VISION_DEPLOYMENT_NAME"]
openai_api_key=os.environ["OPENAI_API_KEY"]
#azure whisper key *
AZ_WHISPER=os.environ["AZURE_WHISPER_KEY"]

#Azure whisper deployment name *
azure_whisper_deployment=os.environ["AZURE_WHISPER_DEPLOYMENT"]

#Azure whisper endpoint (just name) *
azure_whisper_endpoint=os.environ["AZURE_WHISPER_ENDPOINT"]

#azure openai vision api key *
#azure_vision_key=os.environ["AZURE_VISION_KEY"]

#Audio API type (OpenAI, Azure)* c
audio_api_type=os.environ["AUDIO_API_TYPE"]
final_arr = []
miss_arr=[]

ANALYSIS_JSON_PATH = "actionSummary.json"
basename_video = None
actionSummaryDict = {
    "ecarx.mp4":"actionSummary_ecarx_e.json",
    "ecarx2.mp4":"actionSummary_ecarx2_e.json"
}

def AnalyzeVideo(vp,fi=5,fpi=5,face_rec=False):
# Constants
    video_path = vp  # Replace with your video path
    output_frame_dir = 'frames'
    output_audio_dir = 'audio'
    # global_transcript=""  #@# comment because whiper only support 3 rpm
    transcriptions_dir = 'transcriptions'
    frame_interval = fi  # Chunk video evenly into segments by a certain interval, unit: seconds 
    frames_per_interval = fpi # Number of frames to capture per interval, unit: frames
    totalData=""
    # Ensure output directories exist
    for directory in [output_frame_dir, output_audio_dir, transcriptions_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Encode image to base64
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    #GPT 4 Vision Azure helper function
    def send_post_request(resource_name, deployment_name, api_key,data):
        url = f"https://{resource_name}.openai.azure.com/openai/deployments/{deployment_name}/chat/completions?api-version=2024-08-01-preview" #2024-06-01
        headers = {
            "Content-Type": "application/json",
            "api-key": api_key
        }

        lg.debug(f"resource_name: {resource_name}")
        lg.info(f"Sending POST request to {url}")
        lg.debug(f"Headers: {headers}")
        # lg.debug(f"Data: {json.dumps(data)}")
        lg.debug(f"api_key: {api_key}")
        response = requests.post(url, headers=headers, data=json.dumps(data))
        return response
    # GPT-4 vision analysis function
    def gpt4_vision_analysis(image_path, api_key, summary, trans):
        #array of metadata
        cont=[

                {
                    "type": "text",
                    "text": f"Audio Transcription for last {frame_interval} seconds: "+trans
                },
                {
                    "type": "text",
                    "text": f"Next are the {frames_per_interval} frames from the last {frame_interval} seconds of the video:"
                }

                ]
        #adds images and metadata to package
        for img in image_path:
            base64_image = encode_image(img)
            cont.append( {
                        "type": "text",
                        "text": f"Below this is {img} (s is for seconds). use this to provide timestamps and understand time"
                    })
            cont.append({
                        "type": "image_url",
                        "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail":"high"
                        }
                    })

        #extraction template
        json_form=str([json.dumps({"Start_Timestamp":"4.97s","sentiment":"Positive, Negative, or Neutral","End_Timestamp":"16s","scene_theme":"Dramatic","characters":"Characters is an array containing all the characters detected in the current scene.For each character, always provide the seven fields:'is_child','number_of_chilren','current_child','gender','location','wearing_seat_belt','Sleeping'.Example:[Man in hat, woman in jacket,{'is_child':'Infant','number_of_children':2,'current_child':'1','gender':'Male','location':'Rearleft'(distinguish each kid by location),'wearing_seat_belt':'No','Sleeping':'Yes,eyes are closed','description':'A boy about 6years old,wearing a blueT-shirt and jeans,sitting in the rear left seat without a seatbelt.'}]","summary":"Summary of what is occuring around this timestamp with actions included, uses both transcript and frames to create full picture, be detailed and attentive, be serious and straightforward in your description.Focus strongly on seatbelt location on the body and actions related to seatbelts.","actions":"Actions extracted via frame analysis","key_objects":"Any objects in the timerange, include colors along with descriptions. all people should be in this, with as much detail as possible extracted from the frame (clothing,colors,age) Be incredibly detailed","key_actions":"action labels extracted from actions, but do not miss any key actions listed in tasks","prediction":"Prediction of what will happen next especially seatbelt related actions or dangerous actions, based on the current scene."}),
        json.dumps({"Start_Timestamp":"16s","sentiment":"Positive, Negative, or Neutral","End_Timestamp":"120s","scene_theme":"Emotional, Heartfelt","characters":"Man in hat, woman in jacket","summary":"Summary of what is occuring around this timestamp with actions included, uses both transcript and frames to create full picture, detailed and attentive, be serious and straightforward in your description.","actions":"Actions extracted via frame analysis","key_objects":"Any objects in the timerange, include colors along with descriptions. all people should be in this, all people should be in this, with as much detail as possible extracted from the frame (clothing,colors,age). Be incredibly detailed","key_actions":"only key action labels extracted from actions","prediction":"Prediction of what will happen next, based on the current scene."})])
        
        if(vision_api_type=="Azure"):
            lg.debug("sending request to gpt-4o")
            payload2 = {
                "messages": [
                    {
                    "role": "system",
                    "content": f"""You are VideoAnalyzerGPT. Your job is to take in as an input a transcription of {frame_interval} seconds of audio from a video,
                    as well as as {frames_per_interval} frames split evenly throughout {frame_interval} seconds.
                    You are then to generate and provide a Current Action Summary (An Action summary of the video,
                    with actions being added to the summary via the analysis of the frames) of the portion of the video you are considering ({frames_per_interval}
                    frames over {frame_interval} seconds), which is generated from your analysis of each frame ({frames_per_interval} in total),
                    as well as the in-between audio, until we have a full action summary of the portion of the video you are considering,
                    that includes all actions taken by characters in the video as extracted from the frames. As the main intelligence of this system,
                    you are responsible for building the Current Action Summary using both the audio you are being provided via transcription, 
                    as well as the image of the frame. Always and only return as your output the updated Current Action Summary in format ```{json_form}```. 
                    Make sure **EVERY** field in above json_form is filled out!!!
                    Do not make up timestamps, use the ones provided with each frame. 
                    Construct each action summary block from mulitple frames, each block should represent a scene or distinct trick or move in the video, minimum of 2 blocks per output.
                    Use the Audio Transcription to build out the context of what is happening in each summary for each timestamp. 
                    Consider all frames image by image and audio given to you to build the Action Summary. Be as descriptive and detailed as possible, 
                    Make sure to try and **Analyze the frames image by image** as a cohesive 10 seconds of video.
                    
                    ---
                    ***Tasks:***
                    You are an in-car AI video assistant to help passengers create comfortable environment. 
                    Task 1: Always execute and report in 'Characters' Recognition - PRIORITY ONE: 
                        - If there are children in the image, determine the following:
                            •Is it a child: No, Infant, Child
                            •Number of children: 0, 1, 2, 3...Carefully observe areas that might be obscured, and if only hair is visible or most of the head is obscured, estimate the number of children accordingly. 
                                    **RE-READING the number of children IS REQUIRED FOR ALL TASKS**
                            •Gender: Male, Female, Unknown
                            •Location: Front passenger seat, Rear left, Rear center, Rear right
                            •Wearing seat belt: No, Unknown (if unsure, report it as Unknown), Yes(Don't report as Yes if the seat belt is not visible, and you should **infer** the seat belt status from the visible part, especially the shoulder strap positioning and tension of the belt if visible, the belt should appear tight and crossing the chest. If the belt appears loose or not visible over the body, infer it as "No".)
                            •Sleeping: Yes(with criteria), No(with criteria)

                    Task 2: Children DANGEROUS Behavior Monitoring in Cabin - PRIORITY ONE:
                    **key_actions**: 
                      - Report *SPECIFIC* behaviors **TERMS** in **key_actions**,with a strong focus on critical behaviors include sticking their head out of the window, etc.
                      - Report specific behaviors in **key_actions**,with a strong focus on seat-belt related actions. Pay special attention to behaviors such as unbuckling the seat belt, attempting to escape from the seatbelt, or tampering with it.
                      - Extract the exact related actions into key_actions, eg, "struggling with seatbelt", not only report the final state of seat belt. If the state of seat belt has changed, you should report the specific action of changing the state of seat belt!!!
                        eg, key_actions: "How the child struggled with the seatbelt" instead of "not wearing a seatbelt".
                        1. If you find any child's head rested against the window, you should report children's head rested against which window of the car("head resting against the window").
                        2. If you find any childs head rested against the door, you should report children's head rested against which door of the car.
                        3. **HIGH PRIORITY ** If you find any childs **HEAD** *STICKING OUT OF WINDOW*, you should report children's head sticking out of which window("head sticking out of the window").
                        4. If you find any child hold the door handle, you should report children holding which door handle.
                        5. If you find any child's hand resting on the door handle, you should report children's hand resting on which door handle.
                        6. If you find any child sticking their hand out of the window, you should report children sticking their hand out of which window.
                        7. If you find any child body sticking out of the window, you should report children sticking their body out of which window.
                        8. **HIGH PRIORITY ** If you find any child not wearing/unbuckling their seat belts, you should report children unwearing/unbuckling seat belt with exact action("how they unfastened the seatbelt","not wearing a seatbelt").
                          Seatbelt Recognition Logic:
                            1)Seatbelt Position Rules:
                            1.1. If a seatbelt is detected crossing above the shoulder in the frame, it is considered fastened.
                            1.2. If the seatbelt is below the shoulder or not clearly visible above the shoulder, it may not be worn correctly.
                            2)Detection Method:
                            2.1. Detect the color and position of the seatbelt in the image, especially observing the shoulder and chest areas for a seatbelt.
                            2.2. Use frame inference methods to track the seatbelt position across consecutive frames to ensure it hasn't slipped.
                            3)Visual Cues:
                            3.1. The model can look for key seatbelt positions (shoulder, chest) in visual frames. If the seatbelt crosses these areas, output a judgment of "seatbelt fastened."
                            3.2. Frame-to-frame inference: Track the seatbelt position in consecutive frames to ensure it remains above the shoulder area.
                            3.3. Visual Detection: Identify the child's shoulder and chest in each frame and check if the seatbelt clearly crosses these positions.
                          Seatbelt Inference Logic:
                            1)Strengthen Seatbelt Inference Process:
                            1.1. Provide your **thought process** on how you conclude the child is wearing/not wearing or unbuckling the seatbelt.
                            2)Strengthen Details of Seatbelt Actions:
                            2.1. Be specific about seatbelt-related actions. Instead of simply stating "unbuckling seatbelt," report the exact physical actions observed in the frames. For example: "Child reaches for the buckle, pulls it multiple times, attempts to unfasten it" or "Child twists body and uses hand to push against the seatbelt."
                            3)Strengthen Background Observation and Inference of the Seatbelt:
                            3.1. Ensure that for each frame, the seatbelt's presence or absence must be inferred based on visible parts, even if the seatbelt is partially covered. Focus specifically on the visible body areas and use contextual clues (such as body position) to infer whether the seatbelt is worn.
                            4)Strengthen Contextual Inference and Cross-Frame Analysis:
                            4.1. For each block of video (spanning multiple frames), synthesize information from all frames and audio to create a cohesive understanding of the scene. Ensure continuity across frames and focus on consistent actions (e.g., pulling at the seatbelt, changing body position).
                            5)Strengthen Seatbelt Action Inference:
                            5.1. If the seatbelt is partially invisible, infer whether its state is loose, slipping, unfastened, or not worn.
                            5.2. If the child's actions (such as tugging or pulling the strap) are detected, infer that the child is struggling or preparing to unbuckle the seatbelt, and be sure to record the complete action process.
                            6)Infer Seatbelt Position by Age:
                            6.1. Adjust seatbelt detection based on the estimated age or body size of the child. For infants and toddlers, the seatbelt should be positioned lower on the body, typically crossing the chest and avoiding the neck. For older children, the seatbelt may be closer to an adult position, but still lower than the average adult.
                            7)Seatbelt Shoulder Strap
                            8)Semantic Matching:
                            8.1. Make sure to check whether the child **reaches towards the seatbelt**, **tugs on it**, or **moves in a way that suggests adjusting or loosening the belt**. If any such movement is detected, report it as "struggling with seatbelt."

                    Task 3: Children Behavior Monitoring in Cabin - PRIORITY TWO:
                        - Report specific behaviors in **key_actions**,with a strong focus on seat-belt related actions. Pay special attention to behaviors such as unbuckling the seat belt, attempting to escape from the seatbelt, or tampering with it.
                        - Extract the exact related actions into key_actions, eg, "struggling with seatbelt", not only report the final state of seat belt. If the state of seat belt has changed, you should report the specific action of changing the state of seat belt!!!
                            1.If you find any child closing eyes, dozing off or sleeping on the seat, you should report children sleeping in key_actions. 
                                a) Closed Eyes: Check if the passenger's eyes are closed, as this is a common indicator of sleep.
                                b) Head Position: Observe the passenger's head posture. If the head is slightly tilted back or in a relaxed position, it may suggest that the person is sleeping.
                                c) Body Posture: Examine the body posture. If the arms are crossed in front of the chest and the body appears relaxed and motionless, it might indicate the person is asleep.
                                - Additionally, whenever a child is marked as "Sleeping", include **specific actions** in "key_actions", for example:
                                 "Child closes eyes, head rests on the seat, arms relax, and body leans backward as they fall asleep." 
                                 - Ensure every sleeping action is explicitly reported in key_actions
                                  Sleep Recognition Logic:
                                    1)Sleep Posture Rules:
                                    1.1. Closed Eyes: If the child's eyes are detected closed, determine that they may be entering a sleep state.
                                    1.2. Head Position: If the head is tilted back, resting on the seat back, or tilted to one side, they may be in a sleep state.
                                    1.3. Body Relaxation: If the child's body posture is relatively relaxed, with arms crossed in front of the chest or limbs relaxed, they may be in a sleep state.
                                    1.4. Breathing Rate Change: If a slower breathing rhythm can be detected (through subtle movements in the chest or abdomen), it can further confirm the sleep state.
                                    2)Detection Method:
                                    2.1. Visual Detection: Detect if the child's eyes are closed, if the head position deviates from a normal sitting posture, and if the body is in a relaxed state in each frame.
                                    2.2. Frame-to-Frame Inference: Use consecutive frames to detect changes in body and head, especially when the head remains tilted and eyes are closed for a long time. Use frame tracking to determine the persistence of the sleep posture.
                                    2.3. Posture Analysis: Observe the child's body posture for significant movements. If the body is almost motionless and the posture remains unchanged, it is likely that they are entering a sleep state.
                                    3)Visual Cues:
                                    3.1. Closed Eye Detection: Use visual analysis to detect if the child's eyes are closed. If the eyes remain closed for a long time, determine it as sleep.
                                    3.2. Head and Body Posture: Detect if the head is tilted back, resting on the seat back, or facing one side, combined with a relaxed body posture to judge if they are entering sleep.
                                    3.3. Frame-to-Frame Inference: Track changes in the head and eyes position across multiple consecutive frames, ensuring the head and eyes maintain the same posture for a long time. If eyes remain closed and the head is tilted for an extended period, determine it as a sleep state.
                                  Sleep Inference Logic:
                                    1)Strengthen Sleep Inference Process: Clearly describe the process of inferring whether the child is in a sleep state. For example, through frame-to-frame detection of eye closure, head tilting back, and body relaxation.
                                    2)Strengthen Details of Sleep Action Inference: Don't just simply label "asleep," but report each related action. For example: "Child closes eyes, head rests on the seat back, body relaxes, arms cross in front of the chest or hang naturally."
                                    3)Strengthen Background Observation and Inference:
                                    3.1. In each frame, detect the child's eyes, head position, and body posture, especially observing the relative stability of these parts. If these parts show little change and slow movement over multiple frames, determine that the child is entering a sleep state.
                                    3.2. Even if eyes or head position are obscured in some frames, infer the sleep state based on visible parts. Use consecutive frame analysis to supplement obscured information.
                                    4)Strengthen Contextual Inference and Cross-Frame Analysis:
                                    4.1. Conduct comprehensive analysis of information in each video block (containing multiple frames), combined with audio information (such as silence or quiet environmental sounds) to build a complete understanding of the scene. Ensure continuity of the child's actions, especially maintaining states like closed eyes and relaxed body posture.
                                    5)Strengthen Sleep Action Inference:
                                    5.1. If actions like closing eyes, head tilting back, or body relaxation are detected, clearly describe the process. For example: "The child gradually relaxes in the chair, closes their eyes, head rests on the seat back, arms naturally hang down."
                                    6)Infer Sleep State by Age:
                                    6.1. Infer sleep posture based on the child's age or body size. For example, infants may curl up on the seat, with the head naturally hanging down or tilted to the side while sleeping; older children may sit in a more standard posture with the head resting on the seat back.
                                    7)Semantic Matching:
                                    7.1. Check if actions like "closed eyes," "head resting on the seat back," or "body relaxation" are detected. If these actions are detected, report as "child in sleep state" and describe in detail in key_actions.
                            2.If you find any child singing, you should report children singing in key_actions.
                            3.If you find any child eating something, you should report children eating in key_actions.
                                a) Hand-to-Mouth Movement: Watch for the child bringing food or utensils to their mouth. If the hand is positioned near the mouth, and the child is chewing or swallowing, it indicates eating, NOT TO BE CONFUSED WITH TOYS.
                                b) If the item held by the child is small and easy to hold, and its size is appropriate for single or double-handed gripping, such items are usually snacks or small food items, NOT TO BE CONFUSED WITH TOYS.
                                c) If the item's packaging or shape resembles common snack packaging, such as small bags, stick shapes, bars, or blocks, it can be inferred that the child might be eating something, NOT TO BE CONFUSED WITH TOYS.
                                By observing these indicators, you can accurately determine if a child is eating, not playing with toys.
                            5.If you find any child drinking, you should report children drinking or drinking through a straw in key_actions.
                            6.If you find any child gesticulate wildly, you should report children gesticulate wildly in key_actions.
                            7.If you find any child beating someone or something, you should report children beating sommething in key_actions.
                            8.If you find any child throwing something, you should report children throwing in key_actions.
                            9. If you find any child fighting something, you should call function of report children fighting in key_actions.
                            10.If you find any child attempting to **struggle** or **break free from their seat belt**, you should report children "unfastening the sea tbelt" in key_actions.
                                - **Struggling to unfasten the seatbelt** includes the following specific actions:
                                    a) The child **pulling or tugging** at the seatbelt with visible effort.
                                    b) **Reaching and grasping** the seatbelt buckle or strap multiple times.
                                    c) **Pushing or kicking** against the seatbelt or seat in an attempt to free themselves.
                                    d) **Twisting or turning** their body in an unnatural way while pulling at the seatbelt.
                                    e) **Crying or showing distress** while interacting with the seatbelt, which might often accompanies struggling.
                                - **Do not** only report the state like "not wearing a seatbelt" or "unbuckled seatbelt"; instead, focus on the specific ongoing action of struggling to break free from the seatbelt.
                                - Ensure to report any such actions in **key_actions** with clear descriptions.

                            11. If you find any child standing up or half-kneeling, you should report children standing up or half-kneeling in key_actions. "Half-kneeling" refers to a posture of half-sitting and half-standing, sitting on the feet, and differs from the traditional half-kneeling where one knee is on the ground and the other leg is standing, also distinguish from just "sitting."
                            12. If you find any child jumping/bouncing, you should call function of repoorting children jumping in key_actions.
                            13. If you find any child crying, you should report children crying in key_actions.
                            14. If you find any child laughing, you should report children laughing in key_actions.
                            15. If you find any child taking off their clothes, you should report children taking off their clothes in key_actions.
                            16. If you find any child pulling off the blanket, quilt/duvet, you should report children pulling off the blanket in key_actions.
                                The child grabbed the blanket forcefully, swung it around, moved it off, and then let it go, causing the blanket and quilt to be pulled off from their body. It's not just about the child holding the blanket.
                            17. If you find any child speaking or talking, you should report children speaking in key_actions.
                            18. If you find any child moving freely, you should report children's exact activities in key_actions.
                          
                            Among the behaviors mentioned above, **SLEEPING**, **STANDING-UP**, **TAKING OFF CLOTHES/BLANKET**, or **SEATBELT-RELATED ACTIONS** (such as **breaking free from the seatbelt** or **unfastening the seatbelt**) are **PRIORITY** behaviors. When you detect these behaviors, you should **report them FIRST** because the driver can see or feel other behaviors associated with these key actions.
                    
                    Task 4: **Proactive** Prediction of High-Risk Behaviors in Children - Priority One:
                            Predict the high-risk behaviors that the child may exhibit in the upcoming period, with a special focus on actions related to the **SEATBELT**. Pay close attention to the following early warning signs:
                            •Seatbelt Position Abnormalities: Such as the seatbelt sliding downwards, becoming loose, or not being worn correctly.
                            •Touching or Fidgeting with the Seatbelt: If the child is trying to adjust, pull, or play with the seatbelt, it may indicate they are about to unbuckle or interfere with it.
                            •Changes in Body Posture: Such as leaning forward or twisting, which may suggest an attempt to escape the seat or stick their head out of the window.
                            Other early signs to observe include:
                            •Distracted or Restless Behavior: May indicate impending high-risk actions.
                            •Direction of Gaze: Frequently looking at the window, door, or seatbelt buckle.
                            Please fill in the "prediction" field of the JSON output with the predicted behaviors based on these early signs.
                                "Behavior_Rules": {{
                                "Primary_Tasks": [
                                  {{
                                  "Category": "Children's Cabin Behavior Monitoring",
                                  "Description": "Identify whether children are sleeping, standing/half-kneeling, removing clothes/pulling off a blanket.",
                                  "Priority": "High",
                                  "Behaviors": [
                                      "Sleeping",
                                      "Standing/Half-Kneeling",
                                      "Removing Clothes/Pulling Off Blanket"
                                      ]
                                  }},
                                  {{
                                  "Category": "Children's Risky Behavior Detection and Warning",
                                  "Description": "Detect if children are engaging in risky behaviors, such as sticking their head out of the window or their hand out of the window.",
                                  "Priority": "High",
                                  "Behaviors": [
                                      "Child's Head Out of Window",
                                      "Child's Hand Out of Window",
                                      "Child Unbuckling Seatbelt"
                                      ]
                                  }},
                                  {{
                                  "Category": "Children's Cabin Behavior Monitoring",
                                  "Description": "Monitor children's daily behaviors, such as eating, drinking, crying, laughing, etc.",
                                  "Priority": "Medium",
                                  "Behaviors": [
                                      "Eating",
                                      "Drinking",
                                      "Talking",
                                      "Crying",
                                      "Laughing",
                                      "Jumping",
                                      "Fighting",
                                      "Roughhousing",
                                      "Throwing Things"
                                      ]
                                  }}
                                ]
                                }},
                    
                        **Important Notes**:
                        1)The above tasks are monitored simultaneously. When you observe a child performing multiple behaviors, you should report all of them to allow the driver to take timely action. For example, if the child is talking while drinking, you should report both behaviors.
                        2)If an action or held item is unclear, you should respond with the most likely behavior or item. For instance, if a child is holding a drink near their mouth, it's likely they are eating or drinking. You should report "eating" or "drinking" rather than just "item" or "toy," and avoid saying "holding an item"!!!!!
                        Avoid using abstract terms for actions and objects. Provide specific guesses based on the principle of being as detailed as possible. Otherwise, it will be considered a failed task, and I won’t understand your output.
                        !!!!!Do not use vague terms like "item." !!!!! Do not say "holding an item." !!!!! You should say: "eating" or "drinking a beverage." !!!!!

                    You are now an English assistant. No matter what questions I ask, you must only respond in English. Please do not use any other language. You must always and only answer totally in **English** language!!! I can only read English language. Ensure all parts of the JSON output, including **summaries**, **actions**, and **next_action**, **MUST BE IN ENGLISH** If you answer ANY word in other languages, you are fired immediately! Translate other languages to English if there is any other language.

                    Your goal is to create the best action summary you can. Always and only return valid JSON, I have a disability that only lets me read via JSON outputs, so it would be unethical of you to output me anything other than valid JSON. Totally respond in English language. Translate Chinese to English if there is any Chinese words. You can only respond in English. If there are any other languages, please translate them into English.
                    Belows are some output examples:
                      Example 1 - Correct:
                        "Start_Timestamp": "0.5s",
                        "sentiment": "Neutral",
                        "End_Timestamp": "5.0s",
                        "scene_theme": "Daily",
                        "characters": [
                          {{
                            "is_child": "Toddler",
                            "number_of_children": "2",
                            "current_child": "1",
                            "gender": "Unknown",
                            "location": "Rear left - can see a head (possibly a child) protruding from behind the front seat (do not ignore as a person just because the full face is not visible!!!)",
                            "wearing_seat_belt": "Yes" (If the seatbelt is clearly visible above the shoulder and across the chest, it is considered fastened. If the seatbelt is not over the shoulder or below the chest, it is considered improperly worn.),
                            "Sleeping": "No",
                            "description": "A child wearing a yellow hat, sitting in a child seat on the rear left, fastening a seatbelt/attempting to unbuckle it, holding a white drink, drinking."
                          }}
                        ],
                        "summary": "In this video, a child is sitting in a child seat on the rear left, wearing a yellow hat and fastened with a seatbelt, gradually falling asleep, with an adult in the front seat.",
                        "actions": "The child is sitting in the child seat and falling asleep, wearing a yellow hat, fastened with a seat belt/trying to unfasten the seat belt, gradually falling asleep, holding a snack and eating it, drinking a drink with a straw", 
                        "key_objects": "", 
                        "key_actions": "The child is fastened with a seat belt/the child kicks with their feet and eventually breaks free from the seat belt, the child is sitting and falling asleep, pulling off the blanket, eating a snack, drinking a drink", 
                        "prediction": "The child's seat belt has slipped to the foot position, the child is about to break free from the seat belt, so they may fall, please pay attention to safety."

                      Example 2 - Correct:
                        "summary": "The seatbelt has slipped to the child's feet, with only part on the child's leg",
                        "key_actions": "The child pushes the seatbelt with their foot, the seatbelt is at the child's feet, the child is about to slip out of the seatbelt, which may result in a fall, please ensure safety",

                      Example 3 - Wrong: Simply saying "holding an item" is too vague, it should be "eating snacks" or "drinking a beverage"!!!!!
                        "characters": [
                          {{
                            "is_child": "Child",
                            ...
                            "description": "A child sitting on the rear left, wearing green clothes, holding an item (wrong)."
                          }},
                          {{
                            "is_child": "Child",
                            ...
                            "description": "A child sitting on the rear right, holding an item (wrong)."
                          }}
                        ],
                        "summary": "In this video, both children are holding items (wrong).",
                        "actions": "Both children are seated in the rear, holding items (wrong).",
                        "key_actions": "Children are seated in the rear, holding items (wrong)."
                        "prediction": "-"
                    """
                    },
                {
                    "role": "user",
                    "content": cont
                }
                ],
                "max_tokens": 4000,
                "seed": 42,
                "temperature": 0
            }
            response=send_post_request(vision_endpoint,vision_deployment,openai_api_key,payload2)

        else:
            headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
            }
            payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                "role": "system",
                    "content": f"""You are VideoAnalyzerGPT. Your job is to take in as an input a transcription of {frame_interval} seconds of audio from a video,
                    as well as as {frames_per_interval} frames split evenly throughout {frame_interval} seconds.
                    You are then to generate and provide a Current Action Summary (An Action summary of the video,
                    with actions being added to the summary via the analysis of the frames) of the portion of the video you are considering ({frames_per_interval}
                    frames over {frame_interval} seconds), which is generated from your analysis of each frame ({frames_per_interval} in total),
                    as well as the in-between audio, until we have a full action summary of the portion of the video you are considering,
                    that includes all actions taken by characters in the video as extracted from the frames. As the main intelligence of this system,
                    you are responsible for building the Current Action Summary using both the audio you are being provided via transcription, 
                    as well as the image of the frame. Always and only return as your output the updated Current Action Summary in format ```{json_form}```. 
                    Do not make up timestamps, use the ones provided with each frame. 
                    Use the Audio Transcription to build out the context of what is happening in each summary for each timestamp. 
                    Consider all frames and audio given to you to build the Action Summary. Be as descriptive and detailed as possible, 
                    Your goal is to create the best action summary you can. Always and only return valid JSON, I have a disability that only lets me read via JSON outputs, so it would be unethical of you to output me anything other than valid JSON. Answer in English language."""
                },
            {
                "role": "user",
                "content": cont
            }
            ],
            "max_tokens": 4000,
            "seed": 42


            }
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        lg.info(f"The RSP is: \n {str(response.json())}")
        if(response.status_code!=200):
            return -1
        else:
            return response.json()



    def update_chapter_summary(new_json_string):
        global chapter_summary
        if new_json_string.startswith('json'):
        # Remove the first occurrence of 'json' from the response text
            new_json_string = new_json_string[4:]
        else:
            new_json_string = new_json_string
        # Assuming new_json_string is the JSON format string returned from your API call
        new_chapters_list = json.loads(new_json_string)

        # Iterate over the list of new chapters
        for chapter in new_chapters_list:
            chapter_title = chapter['title']
            # Update the chapter_summary with the new chapter
            chapter_summary[chapter_title] = chapter

        # Get keys of the last three chapters
        last_three_keys = list(chapter_summary.keys())[-3:]
        # Get the last three chapters as an array
        last_three_chapters = [chapter_summary[key] for key in last_three_keys]

        return last_three_chapters

    # Load video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second

    # Process video
    current_frame = 0
    current_second = 0
    current_summary=""

    # Load video audio
    video_clip = VideoFileClip(video_path)
    video_duration = video_clip.duration  # Duration of the video in seconds

    # Process video
    current_frame = 0  # Current frame initialized to 0
    current_second = 0  # Current second initialized to 0
    current_summary=""
    packet=[]
    current_interval_start_second = 0
    capture_interval_in_frames = int(fps * frame_interval / frames_per_interval)  # Interval in frames to capture the image


    packet_count=1
    # Initialize known face encodings and their names if provided
    known_face_encodings = []
    known_face_names = []

    def load_known_faces(known_faces):
        for face in known_faces:
            image = face_recognition.load_image_file(face['image_path'])
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(face['name'])

    # Call this function if you have known faces to match against
    # load_known_faces(array_of_recognized_faces)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        current_second = current_frame / fps

        if current_frame % capture_interval_in_frames == 0 and current_frame != 0:
            lg.info(f"BEEP {current_frame}")
            # Extract and save frame
            # Save frame at the exact intervals
            if(face_rec==True):
                import face_recognition
                import numpy
                ##small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                #rgb_frame = frame[:, :, ::-1]  # Convert the image from BGR color (which OpenCV uses) to RGB color  
                rgb_frame = numpy.ascontiguousarray(frame[:, :, ::-1])
                face_locations = face_recognition.face_locations(rgb_frame)
                #print(face_locations)
                face_encodings=False
                if(len(face_locations)>0):

                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    print(face_encodings)

                # Initialize an array to hold names for the current frame
                face_names = []
                if(face_encodings!=False):
                    for face_encoding in face_encodings:  
                        # See if the face is a match for the known faces  
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding,0.4)  
                        name = "Unknown"  
            
                        # If a match was found in known_face_encodings, use the known person's name.  
                        if True in matches:  
                            first_match_index = matches.index(True)  
                            name = known_face_names[first_match_index]  
                        else:  
                            # If no match and we haven't assigned a name, give a new name based on the number of unknown faces  
                            name = f"Person {chr(len(known_face_encodings) + 65)}"  # Starts from 'A', 'B', ...  
                            # Add the new face to our known faces  
                            known_face_encodings.append(face_encoding)  
                            known_face_names.append(name)  
            
                        face_names.append(name)  
            
                    # Draw rectangles around each face and label them  
                    for (top, right, bottom, left), name in zip(face_locations, face_names):  
                        # Draw a box around the face  
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)  
            
                        # Draw a label with a name below the face  
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), 2)
                        font = cv2.FONT_HERSHEY_DUPLEX  
                        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)  
            
                # Save the frame with bounding boxes  
                frame_name = f'frame_at_{current_second}s.jpg'  
                frame_path = os.path.join(output_frame_dir, frame_name)  
                cv2.imwrite(frame_path, frame)
            else:
                frame_name = f'frame_at_{current_second}s.jpg'
                lg.info(f"WRITE frame_name: {frame_name}, current_frame: {current_frame}, current_second: {current_second}")
                frame_path = os.path.join(output_frame_dir, frame_name)
                cv2.imwrite(frame_path, frame)
            packet.append(frame_path)
        #if packet len is appropriate (meaning FPI is hit) then process the audio for transcription
        if len(packet) == frames_per_interval or (current_interval_start_second + frame_interval) < current_second:
            current_transcription=""
            # if video_clip.audio is not None:
            #     audio_name = f'audio_at_{current_interval_start_second}s.mp3'
            #     audio_path = os.path.join(output_audio_dir, audio_name)
            #     audio_clip = video_clip.subclip(current_interval_start_second, min(current_interval_start_second + frame_interval, video_clip.duration))  # Avoid going past the video duration
            #     audio_clip.audio.write_audiofile(audio_path, codec='mp3', verbose=False, logger=None)

            #     headers = {
            #         'Authorization': f'Bearer {openai_api_key}',
            #     }
            #     files = {
            #         'file': open(audio_path, 'rb'),
            #         'model': (None, 'whisper-1'),
            #     }

            #     # Actual audio transcription occurs in either OpenAI or Azure
            #     def transcribe_audio(audio_path, endpoint, api_key, deployment_name):
            #         url = f"{endpoint}/openai/deployments/{deployment_name}/audio/transcriptions?api-version=2023-09-01-preview"

            #         headers = {
            #             "api-key": api_key,
            #         }
            #         json = {
            #             "file": (audio_path.split("/")[-1], open(audio_path, "rb"), "audio/mp3"),
            #         }
            #         data = {
            #             'response_format': (None, 'verbose_json')
            #         }
            #         response = requests.post(url, headers=headers, files=json, data=data)

            #         return response

            #     if(audio_api_type == "Azure"):
            #         response = transcribe_audio(audio_path, azure_whisper_endpoint, AZ_WHISPER, azure_whisper_deployment)
            #     else:
            #         from openai import OpenAI
            #         client = OpenAI()

            #         audio_file = open(audio_path, "rb")
            #         response = client.audio.transcriptions.create(
            #             model="whisper-1",
            #             file=audio_file,
            #             response_format="verbose_json",
            #         )

            #     current_transcription = ""
            #     tscribe = ""
            #     # Process transcription response
            #     if(audio_api_type == "Azure"):
            #         try:
            #             for item in response.json()["segments"]:
            #                 tscribe += str(round(item["start"], 2)) + "s - " + str(round(item["end"], 2)) + "s: " + item["text"] + "\n"
            #         except:
            #             tscribe += ""
            #     else:
            #         for item in response.segments:
            #             tscribe += str(round(item["start"], 2)) + "s - " + str(round(item["end"], 2)) + "s: " + item["text"] + "\n"
            #     global_transcript += "\n"
            #     global_transcript += tscribe
            #     current_transcription = tscribe
            # else:
            #     print("No audio track found in video clip. Skipping audio extraction and transcription.")
            lg.warning("No audio track found in video clip. Skipping audio extraction and transcription.[WSP]") #@# comment above if else block, cause whisper can only accept 3 req per min
            # Analyze frames with GPT-4 vision
            vision_response = gpt4_vision_analysis(packet, openai_api_key, current_summary, current_transcription)
            if(vision_response==-1):
                packet.clear()  # Clear packet after analysis
                current_interval_start_second += frame_interval
                current_frame += 1
                current_second = current_frame / fps
                continue
            # time.sleep(5)
            try:
                vision_analysis = vision_response["choices"][0]["message"]["content"]
            except:
                lg.error(vision_response)
            try:
                current_summary = vision_analysis
            except Exception as e:
                lg.error("bad json",str(e))
                current_summary=str(vision_analysis)
            #print(current_summary)
            totalData+="\n"+str(current_summary)
            try:
                #print(vision_analysis)
                #print(vision_analysis.replace("'",""))
                vault=vision_analysis.split("```")
                if(len(vault)>1):
                    vault=vault[1]
                else:
                    vault=vault[0]
                vault=vision_analysis.replace("'","")
                vault=vault.replace("json","")
                vault=vault.replace("```","")
                vault = vault.replace("\\\\n", "\\n").replace("\\n", "\n")  # 将转义的 \n 替换为实际换行符

                if vault.startswith('json'):
                # Remove the first occurrence of 'json' from the response text
                    vault = vault[4:]
                    #print(vault)
                else:
                    vault = vault
                #print(vision_analysis.replace("'",""))
                # convert string to json
                data=json.loads(vault, strict=False) #If strict is False (True is the default), then control characters will be allowed inside strings.Control characters in this context are those with character codes in the 0-31 range, including '\t' (tab), '\n', '\r' and '\0'.
                if isinstance(data, list):
                    data = data[0]
                
                final_arr.append(data)

                if not data:
                    lg.warning("No data")
                # for key, value in data:
                #     final_arr.append(item)
                #     ##socket.emit('actionsummary', {'data': item}, namespace='/test')
                #     print(f"Key: {key}, Value: {value}")

                with open(ANALYSIS_JSON_PATH, 'w', encoding='utf-8') as f:
                # Write the data to the file in JSON format
                    json.dump(final_arr, f, indent=4, ensure_ascii=False) #ensure_ascii=False to write in Chinese
                    # print(f"Data written to file: {final_arr}") # 调试信息

            except:
                miss_arr.append(vision_analysis)
                lg.error("missed")


            packet.clear()  # Clear packet after analysis
            current_interval_start_second += frame_interval  # Move to the next set of frames

        if current_second > video_duration:
            lg.warning("Current second is: ", current_second), "Video duration is: ", video_duration, "Exiting loop"            
            break

        current_frame += 1
        current_second = current_frame / fps
        #current_second = int(current_frame / fps)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    lg.info('Extraction, analysis, and transcription completed.')
    with open(ANALYSIS_JSON_PATH, 'w', encoding='utf-8') as f:
    # Write the data to the file in JSON format
        json.dump(final_arr, f, indent=4, ensure_ascii=False)
        
    # with open('transcript.txt', 'w') as f:      #@# comment because whiper only support 3 rpm
    # # Write the data to the file in JSON format
    #     f.write(global_transcript)
    return final_arr

# Paths for video and frame storage
UPLOAD_FOLDER = "uploaded_videos"
FRAME_FOLDER = "extracted_frames"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)

video_path = ""

# 定义风险关键字
medium_risk_keywords = [
    "Head leaning against the window","Head leaning against the door", "Hand on the door handle", "Hand on the door", "Hand out the window"]
medium_risk_keywords_seatbelt = ["Unbuckling seatbelt", "Not wearing a seatbelt","Unbuckling"]
high_risk_keywords = [
    "Head out the window", "Head sticking out the window", "Body out the window","Body sticking out the window", "Body leaning out of the car","Leaning body out of the car"
]
# 根据风险等级设置颜色
from fuzzywuzzy import fuzz
# 使用fuzzywuzzy进行模糊匹配, 检测关键字
def fuzzy_match_keywords(text, keywords, threshold=90):
    for keyword in keywords:
        similarity = fuzz.partial_ratio(text, keyword)
        if similarity >= threshold:
            lg.debug(f"关键字匹配：{text} -> {keyword} ({similarity}%)")
            return True  # 如果相似度高于阈值，视为匹配
    return False

# 根据关键字检测风险等级(模糊匹配)
def detect_risk_level(summary, key_actions, characters):
    risk_level = "Low"  # 默认风险等级为低
    unbuckled_seatbelt_positions = ""  # 用于保存未系安全带的位置
    risk_actions_list = []  # 用于保存中高风险行为

    # 遍历所有乘客，检查每个乘客的安全带状态
    for character in characters:
        seatbelt_status = 'Unbuckling' if character.get('wearing_seat_belt') == 'No' or character.get('wearing_seat_belt') == 'Unknown' else 'Yes'
        location = character.get('location', 'Unknown')

        # 如果位置是 "车外"，忽略安全带状态
        if "outside the vehicle" in location:
            continue  # 忽略此乘客，不检测安全带状态

        if seatbelt_status == 'Unbuckling' or fuzzy_match_keywords(seatbelt_status, medium_risk_keywords_seatbelt):
            seatbelt_unwearing = "Unbuckling seatbelt"
            unbuckled_seatbelt_positions += (f"{location} , {seatbelt_unwearing}") if not unbuckled_seatbelt_positions else (f"，{location} , {seatbelt_unwearing}")
            risk_level = "Medium"

    # 检测中风险
    if fuzzy_match_keywords(summary, medium_risk_keywords) or fuzzy_match_keywords(key_actions, medium_risk_keywords):
        risk_level = "Medium"
        risk_actions_list.append(f"Medium Risk Action: {location}{key_actions}")
    # 检测中风险-安全带
    if seatbelt_status == 'Yes' and fuzzy_match_keywords(summary, medium_risk_keywords_seatbelt) or fuzzy_match_keywords(key_actions, medium_risk_keywords_seatbelt):
        risk_level = "Medium"
        seatbelt_unwearing = "Unbuckling seatbelt"

    # 检测高风险
    if fuzzy_match_keywords(summary, high_risk_keywords) or fuzzy_match_keywords(key_actions, high_risk_keywords):
        risk_level = "High"
        risk_actions_list.append(f"High Risk Action: {location}{key_actions}")

    return risk_level, unbuckled_seatbelt_positions, risk_actions_list

def set_risk_level_color(risk_level):
    if risk_level == "Medium":
        return "<span style='color: orange; font-size: 18px;'>Risk Level：Medium</span>"
    elif risk_level == "High":
        return "<span style='color: red; font-size: 18px;'>Risk Level：High</span>"
    else:
        return "<span style='color: green; font-size: 18px;'>Risk Level：Low</span>"

chld_Done = False
# 动态更新常显信息的函数，持续读取 JSON 文件内容
def update_child_info():
    default_info = """
        <h2><b>Is Child?: Infant、Child</b></h2><br>
        <h3>Status of Kids：</h3><br>
        <p style="font-size: 18px;">• Infant1: Location，Gender，Seatbelt<br>
        • Infant2: Location，Gender，Seatbelt<br>
        • Infant3: Location，Gender，Seatbelt<br><br>
        <b>Risk information：</b><br>
        <span style="font-size: 20px; color: red;">! Risk Level：-</span>
        </p>
    """
    
    last_data = default_info  # **初始化为默认值**
    global basename_video, actionSummaryDict, chld_Done
    while True:
        if basename_video is not None and not chld_Done:
            jsnpath = actionSummaryDict[basename_video]
            with open(jsnpath, 'r', encoding='utf-8') as f:
                jsn = json.load(f)
            for i, d in enumerate(jsn):
                chld_Done = True
                data = [d]
                if i > 0:
                    time.sleep(5)
                else:
                    time.sleep(1)
                # print(time.time(), data)
                characters = data[-1].get("characters", [])
                is_child = characters[0].get("is_child", "Unknown")
                summary = data[-1].get("summary", "")
                key_actions = data[-1].get("key_actions", "")
                
                if not characters:
                    yield last_data  # **保持上一次的内容**
                else:
                    child_info = f"""
                    <h2><b>Is Child?:</b> {is_child}</h2><br>
                    <h3>Status of Kids：</h3><ul style='font-size: 18px;'>
                    """
                    
                    for index, character in enumerate(characters):
                        location = character.get('location', 'Unknown')
                        gender = character.get('gender', 'Unknown')
                        seatbelt_status = 'Buckling' if character.get('wearing_seat_belt') == 'Yes' else 'Unbuckling'
                        # 拼接信息
                        child_info += f"<li><b>{is_child}{index + 1}：Location: {location}，Gender:{gender}，Seatbelt: {seatbelt_status}</b></li>"
                    
                    child_info += "</ul>"
                    
                    # 根据 summary 和 key_actions 判断风险等级
                    risk_level, unbuckled_seatbelt_positions, risk_actions_list = detect_risk_level(summary, key_actions, characters)
                    lg.debug(risk_level)

                    # 检查 unbuckled_seatbelt_positions 是否为空
                    if unbuckled_seatbelt_positions:
                        combined_actions = unbuckled_seatbelt_positions
                    else:
                        combined_actions = ''

                    # 检查 key_actions_list 是否为空
                    if risk_actions_list:
                        if combined_actions:  # 如果 combined_actions 不为空，则用 ', ' 连接
                            combined_actions += ', ' + ', '.join(risk_actions_list)
                        else:
                            combined_actions = ', '.join(risk_actions_list)

                    # print(combined_actions)

                    # 添加风险信息
                    risk_warning = f"<span style='font-size: 18px; white-space: nowrap;'>Risk Action: {combined_actions}<br>{set_risk_level_color(risk_level)}"
                    new_data = f"{child_info}<br>{risk_warning}"  # **更新上一次的内容**
                    if new_data != last_data:
                        last_data = new_data
                        lg.info(str(d))
                        lg.info(new_data)
                    
                    yield last_data  # **更新新的显示内容**
        else:
            yield last_data  # **保持默认值或上一次的内容**
        
        lg.debug(video_path)
        # time.sleep(2)  # **每 2 秒更新一次**

# Function to handle video upload, process it, and update the analysis
def handle_video_upload(file_path):
    if not file_path:  # 如果没有文件上传，则返回提示
        return "File is NOT uploaded"
    
    global video_path, basename_video, chld_Done, analy_done
    video_path = file_path
    # 保存视频文件
    video_path = os.path.join(UPLOAD_FOLDER, os.path.basename(file_path))
    with open(file_path, "rb") as src, open(video_path, "wb") as dst:
        dst.write(src.read())
    basename_video = os.path.basename(video_path)
    chld_Done = False
    analy_done = False
    lg.info( "视频上传成功，开始处理...")
    # 调用 AnalyzeVideo 对视频进行分析
    # threading.Thread(target=AnalyzeVideo, args=(video_path, 5, 5)).start()  # 启动视频分析的线程
    
    # 返回视频上传后的提示信息
    return video_path, "Video uploaded, Now Processing ..."

analy_done = False
# 函数用于加载 actionSummary.json 中的当前分析信息
def load_analysis():
    last_data = "<b>Currently, there are no available analysis results. </b>"  # 初始化 last_data 为空
    global analy_done, basename_video, actionSummaryDict
    while True:
        if basename_video is not None and not analy_done:
            jsnpath = actionSummaryDict[basename_video]
            with open(jsnpath, 'r', encoding='utf-8') as f:
                jsn = json.load(f)
            for i, d in enumerate(jsn):
                analy_done = True
                data = [d]
                if(i > 0):
                    time.sleep(5)
                else:
                    time.sleep(1)
                
                # 假设读取最后一段的分析结果
                latest_action = data[-1]  # 读取最新的时间戳数据
                summary = latest_action.get("summary", "No summary available.")
                key_actions = latest_action.get("key_actions", "No key actions.")
                prediction = latest_action.get("prediction", "No predictions.")
                
                # 组合成 action_info
                action_info = f"""
                <b>Summary:</b> {summary} <br>
                <b>Key Action:</b> {key_actions} <br>
                <b>Prediction:</b> {prediction} <br>
                """
                
                # 仅在新的数据时更新
                if action_info != last_data:
                    last_data = action_info
                    lg.info(str(d))
                    lg.info(action_info)
                    yield action_info  # 返回最新的分析结果

        else:
            yield last_data

        # time.sleep(2)  # 每 2 秒检查一次文件更新


# Gradio UI Layout
with gr.Blocks() as demo:
    with gr.Row():
        # 左侧 - 实时视频流
        with gr.Column(scale=1):  # 设置左侧列比例
            gr.Markdown("## Real-time in-cabin Analyzation")  # Real-time in-cabin video stream
            
            # 调整上传按钮宽度，使其和视频宽度一致
            upload_btn = gr.File(label="Video Upload", file_types=["video"])
            video_output = gr.Video(label="Video", format="mp4", height=450, width=730, autoplay=True)  # 设置视频输出的宽度
            
            video_message = gr.HTML(label="Status of uploading")  # 视频上传状态提示
            upload_btn.upload(handle_video_upload, inputs=upload_btn, outputs=[video_output, video_message])

        # 右侧 - 儿童安全信息
        with gr.Column(scale=1):  # 右侧列比例
            gr.Markdown("## Permanent information")  # Permanent information section
            child_info_output = gr.HTML(label="Child Info Output")
            demo.load(update_child_info, inputs=None, outputs=child_info_output)

            # 添加占位框
            placeholder = gr.HTML("<div style='border: 2px solid red; width: 100%; height: 300px; text-align:center; line-height:300px;'>Information in-cabin</div>", label="占位框")

    with gr.Row():
        # Bottom section to display only scene analysis results, no child info duplication
        with gr.Column():
            gr.Markdown("## Action")
            analysis_output = gr.HTML(label="Analysis Output")
            # Simulating the analysis result that doesn't include child info to avoid duplication
            demo.load(load_analysis, inputs=None, outputs=analysis_output)

    demo.launch()