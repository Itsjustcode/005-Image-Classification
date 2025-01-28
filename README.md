AI Image Processing and Classification Project


# **Final Report**

---

## **1. Findings from Each Part of the Project**
- **Classifier with Grad-CAM Heatmap:**  
  The classifier identified the image of a dog and provided its top-3 predictions, such as "Golden Retriever" or "Irish Setter." Using Grad-CAM, we generated a heatmap to see which parts of the image the AI focused on most. For example, the heatmap highlighted the dog's body and face as important for its prediction. This taught me how the AI "sees" and prioritizes parts of an image when making decisions.

- **Occlusions (Blocking Parts of the Image):**  
  When we added black boxes or blurred specific parts of the image based on the heatmap, the classifier sometimes struggled to recognize the dog correctly. For instance, when we covered areas the AI focused on (like the dog's face), its confidence dropped significantly. This showed us how sensitive the AI is to losing key details in an image.

- **Filters:**  
  We experimented with creative filters to change how the image looked. Filters like **"solarize"** made the image look inverted and surreal, **"posterize"** simplified the colors to give it a cartoonish look, and **"color enhancement"** made the colors pop by increasing their brightness and vibrancy. Additionally, we created a **"spaghetti-like" filter** that added contour effects, turning the image into an abstract artwork.

---

## **2. What I Learned About the Classifier**
The heatmap and occlusion experiments taught me that AI doesn’t understand images like humans do. Instead, it focuses on patterns, textures, and colors in specific areas. If those areas are blocked or changed, the AI can easily get confused. **Grad-CAM is an amazing tool for visualizing what the AI is “thinking.”**

---

## **3. Reflection on the Filters**
The filters were both fun and insightful. Each filter added a unique artistic touch to the original image. The **"spaghetti-like" filter** was my favorite because it transformed the photo into something abstract, showing how creative programming can be. In this case, almost as it was hand drawn with a pencil. 

---

## **4. My Experience Working with AI**
Working with AI to write Python code was a mix of learning and trial-and-error. At first, I made mistakes, like forgetting to save my script or missing small details, but the AI helped guide me through debugging and improving my work. It explained complicated code in simple ways, which made it easier to understand and made me feel more confident about coding. **Overall, it felt like working with a really smart teacher who had endless patience! 😊**
