"""
Extract metadata (generate summaries) from the Ladder document.
The main function will not extract metadata; you need to manually run this code after modifying the document. Otherwise, the document and metadata may become inconsistent.
"""


import json
import ollama

def METADATA_EXTRACT_TEMPLATE(title, content):
    return [{"role": "user", "content": 
f"""You are an expert technical writer. Your task is to summarize the following section in a concise and structured manner.  

### Instructions:  
- Provide an explanation of what this section is about.  
- Maintain a **clear and structured** format, using bullet points if necessary.  
- Keep the summary **strictly with about 250 words**.  
- Preserve **important technical details** while removing unnecessary information.  
- Ensure the summary is **coherent and well-organized** for easy readability.  

---

### Example:  
* Section Title: What are Ladder and NOM?
* Content:
Ladder and NOM are two complementary frameworks designed to simplify neural network training and analysis, making it more accessible to both beginners and experienced researchers. Together, they form a powerful ecosystem that supports model building, visualization, and training in an intuitive and user-friendly manner.  

**Ladder: A Graphical Interface for Neural Network Development**  
Ladder is a graphical user interface (GUI) designed for building, managing, and visualizing neural network models in an intuitive way. Instead of writing complex code, users can construct models by dragging and connecting various components in a visual workspace. This approach makes Ladder an excellent tool for:

  * Beginners: Provides an easy entry point into neural network development without requiring extensive programming knowledge.
  * Research projects: Facilitates experimentation with different architectures and configurations in a structured visual format.
  * Data analysis: Helps in understanding model structures and performance through clear visual representations.
  * Educational purposes: Offers an interactive way for students and instructors to learn and teach deep learning concepts.

**NOM: A Python-Based Neural Network Library**  
NOM is a Python library built on top of TensorFlow, designed to streamline neural network development by providing a simplified, easy-to-read API. Unlike raw TensorFlow, which requires extensive coding knowledge and complex configurations, NOM abstracts many low-level details, allowing users to focus on model design and experimentation.
Key features of NOM include:

  * Configuration-Based Modeling: Users can define models using structured configurations instead of writing extensive code.
  * Seamless Integration with Ladder: NOM interprets the visual model built in Ladder and converts it into TensorFlow-compatible code.
  * Simplified API for Training and Inference: Users can easily collect data, build model architectures, configure hyperparameters, and perform training and predictions without dealing with TensorFlow’s complexity.
  * Framework Independence: While NOM is currently built on TensorFlow, its modular structure allows adaptation to other deep learning libraries, enabling performance comparisons and alternative implementations.

**How Ladder and NOM Work Together**  
When used together, Ladder and NOM create a seamless workflow for neural network development:

1. Model Creation in Ladder: Users visually construct a neural network model using Ladder’s drag-and-drop interface.
2. Data Conversion by NOM: NOM interprets the graphical model and translates it into a structured, TensorFlow-compatible format.
3. Training and Execution in NOM: The converted model is trained using NOM’s streamlined API, leveraging TensorFlow’s powerful deep learning capabilities.
4. Results Visualization in Ladder: Once training is complete, users can view model performance, loss curves, and predictions through Ladder’s visual interface.

By combining Ladder’s graphical capabilities with NOM’s streamlined Python API, this framework reduces the complexity of neural network development, making deep learning more accessible to a wider audience.

* Summary: 
**Ladder and NOM** are complementary frameworks designed to simplify neural network training and analysis. They provide an intuitive workflow for both beginners and researchers by integrating graphical model building with streamlined Python-based execution.  

#### **Ladder: A Graphical Interface for Neural Networks**  
Ladder offers a **drag-and-drop** GUI for designing and visualizing neural networks, removing the need for complex coding. It is particularly useful for:  
- **Beginners** – Simplifies model creation without programming expertise.  
- **Research Projects** – Facilitates architectural experimentation.  
- **Data Analysis** – Provides clear model structure visualization.  
- **Education** – Enhances interactive learning of deep learning concepts.  

#### **NOM: A Python-Based Neural Network Library**  
NOM is a **TensorFlow-based** Python library that abstracts low-level details, allowing users to define models with structured configurations. Its key features include:  
- **Configuration-Based Modeling** – Reduces extensive coding needs.  
- **Seamless Ladder Integration** – Converts Ladder’s visual models into TensorFlow-compatible code.  
- **Simplified Training & Inference** – Streamlines hyperparameter tuning and execution.  
- **Framework Independence** – Can adapt to other deep learning libraries.  

#### **How They Work Together**  
1. **Model Creation** – Users design neural networks in Ladder.  
2. **Translation** – NOM converts the graphical model into TensorFlow code.  
3. **Training & Execution** – NOM trains the model efficiently.  
4. **Visualization** – Ladder displays results for analysis.  

This integration makes deep learning development more accessible, combining **visual modeling** with **powerful Python-based execution**.

---

Now, generate a well-structured summary following the instructions above. Note: Give me only the content of summary, do not write the section and title again.

### Section Title: "{title}"  
### Content:  
{content}  
"""}]


def extract_sections(document: str):
    sections = document.strip().split("### ")
    parsed_sections = []
    
    for idx, section in enumerate(sections):
        lines = section.strip().split("\n", 1)
        if len(lines) < 2:
            continue
        title, content = lines[0].strip(), lines[1].strip()
        parsed_sections.append({"section": str(idx), "title": title, "content": content})
    
    print(f"Total paragraphs found: {len(parsed_sections)}")
    return parsed_sections

def summarize_section(title: str, content: str, section_num: str, total_sections: int):
    print(f"Processing paragraph {section_num}/{total_sections}: {title}")
    response = ollama.chat(model="ladder_llama3.1", messages=METADATA_EXTRACT_TEMPLATE(title, content))
    return response['message']['content']

def generate_metadata(document: str):
    sections = extract_sections(document)
    metadata = []
    total_sections = len(sections)
    
    for section in sections:
        summary = summarize_section(section["title"], section["content"], section["section"], total_sections)
        metadata.append({
            "category": "Document",
            "section": section["section"],
            "title": section["title"],
            "summary": summary
        })
    
    return json.dumps(metadata, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # Read document from file
    with open("docs/Ladder_RAG_document.md", "r", encoding="utf-8") as file:
        document_text = file.read()

    # Generate metadata
    metadata_json = generate_metadata(document_text)

    # Save metadata to JSON file
    with open("docs/metadata.json", "w", encoding="utf-8") as file:
        file.write(metadata_json)

    print("Metadata extraction complete. Saved to metadata.json")
