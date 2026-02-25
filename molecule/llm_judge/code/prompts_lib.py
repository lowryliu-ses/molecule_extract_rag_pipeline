import os
import re
from functools import partial


class PromptTemplate:
    """# Example usage
    template = "Hello {name}, welcome to {place}. Today is {day}."

    # Create an instance of TemplateProcessor with the template string
    template = PromptTemplate(template_string=template)

    # Call the method to extract the placeholders
    placeholders = template.placeholders

    # Print the list of placeholders
    print(placeholders)

    print(template.format({'day':'d1','name':'n1','place':'p1'}))"""

    def __init__(self, template_file: str = None, template_string: str = None):

        # Initialize the object with a string attribute or a filename
        if template_string:
            self.template = template_string
        elif template_file:
            with open(template_file, "r") as file:
                self.template = file.read()
        else:
            raise Exception(
                "Please provide the template as a string or a path to a txt file"
            )
        self.extract_placeholders()
        self.var_memory = {}
        self.formatted_template = None

    def extract_placeholders(self):
        # Find all words inside {}
        self.placeholders = list(set(re.findall(r"{(.*?)}", self.template)))

    def check_provided_values(self, values):
        return set(values.keys()) == set(self.placeholders)

    def check_provided_values_partial(self, values):
        return set(values.keys()).issubset(set(self.placeholders))

    def format_partial(self, **values):
        if self.check_provided_values_partial(values):
            self.var_memory.update(**values)
        else:
            raise Exception(
                f"Some of the provided values '{list(values.keys())}' "
                f"were not recognized. The chosen prompt needs the following values: '{list(self.placeholders)}'"
            )

    def format(self, **values):
        self.format_partial(**values)
        if self.check_provided_values(self.var_memory):
            self.strip()
            self.formatted_template = self.template.format(**self.var_memory)
            return self.formatted_template
        else:
            raise Exception(
                f"Not all placeholders were provided. You provided values for '{list(self.var_memory)}' "
                f"but the chosen prompt needs '{list(self.placeholders)}'"
            )

    def strip(self):
        self.var_memory = {
            k: v.strip() if isinstance(v, str) else v
            for k, v in self.var_memory.items()
        }


def extract_text_by_tag(
    xml_string: str,
    tag: str,
    selection="last",
):
    """
    Extracts the text inside the specified XML tag and all text outside the tag.

    Args:
        xml_string (str): The XML string to parse.
        tag (str): The name of the XML tag to extract text from.

    Returns:
        tuple: A tuple with two strings:
            1. Text inside the specified tags.
            2. Text outside the tags.
    """

    # Check input arguments
    assert selection in ("first", "last", "all")

    # Regular expression to match the content inside the given tag and the content outside the tag
    tag_pattern = f"<{tag}.*?>(.*?)</{tag}>"

    # Find all content inside the specified tag
    inside_tag_content = re.findall(tag_pattern, xml_string, re.DOTALL)

    # Extract the content outside the tags by removing the matched tags
    # tag_pattern = f"<{tag}.*?>.*?</{tag}>"
    outside_tag_content = re.sub(tag_pattern, "", xml_string, flags=re.DOTALL)

    # Join the inside content into a single string (since multiple occurrences can be found)
    if len(inside_tag_content) > 0:  # Check that there were matches
        if selection == "last":
            inside_tag_text = inside_tag_content[-1]
        elif selection == "first":
            inside_tag_text = inside_tag_content[0]
        else:
            inside_tag_text = " ".join(inside_tag_content)
    else:
        inside_tag_text = ""

    return inside_tag_text, outside_tag_content


def add_tags(
    input_prompt="../prompts/qar_answer.txt",
    save_new_prompt=False,
    output_prompt=None,
):

    # Function to replace each match with the desired format
    def replacement(match):
        word = match.group(1)
        return f"<{word}>{{{word}}}</{word}>"

    # Open and read the content of the input text file
    with open(input_prompt, "r") as file:
        text = file.read()

    # Regular expression to find words enclosed in curly braces
    pattern = r"\{(\w+)\}"

    # Replace all occurrences of words in curly braces using the regex and replacement function
    modified_text = re.sub(pattern, replacement, text)

    if save_new_prompt:
        if output_prompt is None:
            base_path, extension = os.path.splitext(input_prompt)
            output_prompt = base_path + "_with_tags" + extension

        # Write the modified text to the output file
        with open(output_prompt, "w") as file:
            file.write(modified_text.strip())

    return
