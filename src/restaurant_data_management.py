from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
import json
import os
import shutil
import io
import unittest
from unittest.mock import patch

FILEPATH = 'structured_restaurant_data.json'
BACKUP_PATH = 'structured_restaurant_data.json.bak'
EXAMPLE_RESTAURANT_PARAGRAPH = 'Down in **Santa Monica**, **Mar de Cortez** serves as a **sun-drenched**, **casual taqueria** specializing in **Baja-style seafood**. With a **4.2/5** rating, it captures the salt-air energy of the coast through its signature beer-battered snapper tacos and zesty octopus ceviche, making it a premier spot for open-air dining near the pier. Price range: $' #use the second restaurant paragraph as the example
EXAMPLE_OUTPUT = """
    {{
    "name": "Mar de Cortez",
    "location": "Santa Monica",
    "type": "casual taqueria",
    "food_style": "Baja-style seafood",
    "rating": 4.2,
    "price_range": 1,
    "signatures": [
        "beer-battered snapper tacos",
        "zesty octopus ceviche"
    ],
    "vibe": "salt-air energy",
    "environment": "a premier sun-drenched spot for open-air dining near the pier."
    "shortcomings": []
    }}
"""

## Exercise 1: Integrate the LLM model from Lesson 1

# You will need the LLMs you defined in lesson 1 to structure new restaurant paragraph inputs. In addition to these functions, you will need to implement a new function `new_data_entry_process(paragraph, itemId)`, which takes inputs:

# -   `paragraph`: the new restaurant paragraph;
    
# -   `itemId`: the ID of this new item.
    

# This new function combines and uses the generative models you defined in lesson 1 to structure a given new restaurant paragraph.

# In your `restaurant_data_management.py`, copy and paste the following code block and complete the functions.

#   **Important**: Take a screenshot of your implementation of the `new_data_entry_process()` and name it `M1L3_new_data_entry_process.jpg`.

# ```python

#Update your restaurant_data_structure_prompt_generation
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional

### 3.1. Define the schema
class Restaurant(BaseModel):
    name: str
    location: str
    type: str
    food_style: str
    rating: Optional[float] = None
    price_range: Optional[int] = None
    signatures: List[str] = Field(default_factory=list)
    vibe: Optional[str] = None
    environment: str
    shortcomings: List[str] = Field(default_factory=list)

def restaurant_data_structure_prompt_generation(restaurant_paragraph):
    base_system_msg = f"""
    You are a data extraction assistant. Extract data as per given instructions and format
    """
    
    base_user_prompt = f"""
    Task:
    Extract data from free form restraunt data to a structured json representation.
    DO NOT output anything other than json

    OUTPUT JSON FORMAT DESCRIPTIONS
    {{
    "name": "name of the restraunt",
    "location": "location of restraunt",
    "type": "restraunt type",
    "food_style": "style of food/cuisines served",
    "rating": numeric rating of the restraunt out of 5,
    "price_range": convert the number of dollar($) to a integer representation to represent the price range,
    "signatures": [
        "signature dish 1 at the restraunt",
        "signature dish 2 at the restraunt"
    ],
    "vibe": "capture the amdience and essence of the place",
    "environment": "describe the restraunt in one line"
    "shortcomings": capture an array of negetive feedback or shortcomings of the restraunt if any, else empty array []
    }}

    Restaurant description:
    {restaurant_paragraph}

    Example:
    Input Restaurant Description: {EXAMPLE_RESTAURANT_PARAGRAPH}
    Output:
    {EXAMPLE_OUTPUT}
    
    """
    return base_system_msg, base_user_prompt

# Might need to explain why we are using granite here (cheap)
def llm_model(system_msg, prompt_txt, params=None):
    #system_msg: the system message given to the LLM
    #prompt_txt: the user prompt
    
    model_id = "ibm/granite-4-h-small"

    project_id="skills-network"

    credentials = Credentials(
                    url = "https://us-south.ml.cloud.ibm.com"
                    )

    ### 1.1: Define the model by ModelInference
    # parameters = {
    #     GenParams.MAX_NEW_TOKENS: 1024,  # this controls the maximum number of tokens in the generated output
    #     GenParams.TEMPERATURE: 0.2, # this randomness or creativity of the model's responses
    # }
    parameters = {"max_tokens": 1024}
    model = ModelInference(
            model_id=model_id,
            params=parameters,
            credentials=credentials,
            project_id=project_id
        )

    ### 1.2: Define the messages
    prompt = system_msg + "\n" + prompt_txt
    response = model.chat(
        messages=[
            {
                "role": "user",
                "content": prompt  # Ensure content is a string
            }
        ]
    )
    

    ### 1.3: Get the final response output and return it
    return response['choices'][0]['message']['content'].strip()

def JSON_auto_repair_prompts(response, error_message):
    auto_repair_system_msg = """
    You are a senior data extraction assistant. Correct the data as per given instructions and format
    """
    auto_repair_prompt = f"""
    Task:
    There was an error while formatting the json created by your junior data extraction assistant. Review his work and help him correct the data to a structured json representation.
    DO NOT output anything other than json
    Validate as per below descriptions and error message provided, then output the correct JSON

    OUTPUT JSON FORMAT DESCRIPTIONS
    {{
    "name": "name of the restraunt",
    "location": "location of restraunt",
    "type": "restraunt type",
    "food_style": "style of food/cuisines served",
    "rating": numeric rating of the restraunt out of 5,
    "price_range": convert the number of dollar($) to a integer representation to represent the price range,
    "signatures": [
        "signature dish 1 at the restraunt",
        "signature dish 2 at the restraunt"
    ],
    "vibe": "capture the amdience and essence of the place",
    "environment": "describe the restraunt in one line"
    "shortcomings": capture an array of negetive feedback or shortcomings of the restraunt if any, else empty array []
    }}

    Restaurant JSON created by Junior:
    {response}

    JSON validation error:
    {error_message}

    Example Output:
    {EXAMPLE_OUTPUT}
    """

def new_data_entry_process(paragraph, itemId):
    base_system_msg, base_user_prompt = restaurant_data_structure_prompt_generation(restaurant_paragraph=paragraph)
    restaurant_data = llm_model(system_msg=base_system_msg, prompt_txt=base_user_prompt)

    ### 2.2: Validation and Auto Correction loop on the output (Hint: while loop)
    valid = False
    retries = 0
    error_message = ""
    while not valid and retries < 3:
        retries = retries + 1
        try:
            restaurant_data_model = Restaurant.model_validate_json(restaurant_data)
            valid = True
            error_message = ""
            # print(f"Success! Validated: {final_restaurant_data.name}")
        except ValidationError as e:
            valid = False
            error_message = e.json()
            # print(f"Validation failed: {error_message}")

        if not valid:
            base_system_msg, base_user_prompt = JSON_auto_repair_prompts(candidate_json_output=restaurant_data, error_message=error_message)
            restaurant_data = llm_model(system_msg=base_system_msg, prompt_txt=base_user_prompt)
    
    ### 2.3: Append your finalized response to the structured_restaurant_lists
    restaurant_data_json = json.loads(restaurant_data)
    restaurant_data_json['itemId'] = itemId
    # structured_restaurant_lists.append(restaurant_data_json)
    return restaurant_data_json

def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def save_data(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def show_restaurant_card(res, index):
    print(res)

def manage_restaurants(file_path, backup_path):
    while True:
        data = load_data(file_path)
        print(f"\n🏨 RESTAURANT DATABASE | Records: {len(data)}")
        print("1. Browse All (Names)")
        print("2. View Detailed Record")
        print("3. Add New Restaurant")
        print("4. Edit Restaurant Info")
        print("5. Delete Restaurant")
        print("6. Exit")
        
        choice = input("\nAction: ")

        if choice == '1':
            print("\n--- Current Listings ---")
            # YOUR CODE HERE
            # Instruction: 
            # Iterate through the records in the data file and show their names. 
            # If name doesn't exist, print 'N/A'.
            for i in range(len(data)):
                print(data[i]['name'] or 'N/A')

        elif choice == '2':
            # YOUR CODE HERE:
            # Instruction: 
            # Get the record index in demand from the user with input(). Check 
            # the validity of the input index. If the index is valid, use the  
            # helper function show_restaurant_card(res, index); Otherwise, 
            # print "invalid index." 
            index = int(input("Enter the record index: "))
            try:
                res = data[index]
                show_restaurant_card(res, index)
            except IndexError:
                print("invalid index.")

        elif choice in ['3', '4', '5']:
            # Strict Security Warning
            print("\n❗ SECURITY WARNING: You are entering write-mode.")
            print("Changes will be saved to the database immediately.")
            confirm = input("Are you sure? (type 'yes' to proceed): ").lower()
            if confirm != 'yes':
                print("Operation cancelled.")
                continue

            if choice == '3': # ADD NEW DATA
                itemId = 1000000 + len(data) + 1 #the item id for the new data
                
                # YOUR CODE HERE
                # Instruction:
                # First: ask the user to input a new restaurant description.
                # Second: use new_data_entry_process() to process the new paragraph.
                # Third: append the new restaurant data to the original data list.
                # Finally: save using save_data().
                new_description = input("Enter a new restaurant description: ")
                new_restaurant = new_data_entry_process(new_description, itemId)
                data.append(new_restaurant)
                save_data(file_path, data)
                print("✅ Restaurant added.")

            elif choice == '4': # EDIT DATA
                # YOUR CODE HERE
                # Instruction:
                # First: ask for the input record index.
                # Second: iterate over the keys of the current record, and ask for
                #         new values. If the user doesn't want to update, a simple 
                #         Enter can skip. Update it only when the input index is 
                #         valid.
                # Third: save with save_data() and notify ("✅ Record updated.")
                index = int(input("Enter the record index you want to update: "))
                try:
                    restaurant_data = data[index]
                    for key, value in restaurant_data.items():
                        updated_value = input(f"Enter new value for {key}: ")
                        if updated_value:
                            original_type = type(restaurant_data[key])
                            restaurant_data[key] = original_type(updated_value)
                    save_data(file_path, data)
                    print("✅ Record updated.")
                except IndexError:
                    print("invalid index.")

            elif choice == '5': # DELETE DATA
                # YOUR CODE HERE
                # Instruction:
                # First: ask for the input record index.
                # Second: use pop() to delete if the index is valid.
                # Third: save_data() and notify.
                index = int(input("Enter the record index you want to delete: "))
                try:
                    restaurant_data = data.pop(index)
                    save_data(file_path, data)
                    print("✅ Record deleted.")
                except IndexError:
                    print("invalid index.")

        elif choice == '6': # EXIT
            break
        else:
            print("Invalid input.")

class TestRestaurantDatabase(unittest.TestCase):
    
    def setUp(self):
        """Create a temporary clean database for testing."""
        self.test_file = 'structured_restaurant_data_unit_test.json'
        self.test_file_backup = 'structured_restaurant_data_unit_test.json.bak'
        self.initial_data = [{"name": "Test Cafe", "location": "Test City"}]
        with open(self.test_file, 'w') as f:
            json.dump(self.initial_data, f)

    def tearDown(self):
        """Clean up the test file after tests."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        if os.path.exists(self.test_file_backup):
            os.remove(self.test_file_backup)

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_add_and_delete_restaurant_success(self, mock_stdout, mock_input):
        """
        Test Scenario: Add a new restaurant.
        Inputs: '3' (Add), 'yes' (Confirm), 'New Burger Joint', '6' (Exit)
        """
        # We mock the sequence of user inputs
        mock_restaurant = 'The Copper Sprout is a high-concept, Modern Appalachian farm-to-table destination that blends an industrial-chic aesthetic with rustic forest charm, featuring reclaimed wood and amber lighting to create a sophisticated yet cozy vibe. Priced in the $$ category, the menu celebrates seasonal foraging and local heritage, headlined by signature dishes like Cast-Iron Smoked Trout with pickled fiddlehead ferns and hand-foraged Wild Mushroom Risotto with aged goat cheese. The experience is designed to be intimate and earthy, making it a premier spot for those seeking high-quality, smokehouse-influenced cuisine in a refined, atmospheric setting.'
        mock_input.side_effect = ['3', 'yes', mock_restaurant, '6']
        
        # Run the app
        try:
            manage_restaurants(self.test_file, self.test_file_backup)
        except SystemExit:
            pass # Handle exit if your script uses sys.exit()

        # Check if the data was actually saved
        with open(self.test_file, 'r') as f:
            data = json.load(f)
        
        print(data)
        self.assertEqual(len(data), 2)
        self.assertIn("✅ Restaurant added.", mock_stdout.getvalue())

        mock_input.side_effect = ['5', 'yes', 1, '6']
        
        # Run the app
        try:
            manage_restaurants(self.test_file, self.test_file_backup)
        except SystemExit:
            pass # Handle exit if your script uses sys.exit()

        # Check if the data was actually saved
        with open(self.test_file, 'r') as f:
            data = json.load(f)
        
        print(data)
        self.assertEqual(len(data), 1)

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_delete_security_cancel(self, mock_stdout, mock_input):
        """
        Test Scenario: Try to delete but say 'no' to security warning.
        Inputs: '5' (Delete), 'no' (Cancel), '6' (Exit)
        """
        mock_input.side_effect = ['5', 'no', '6']
        
        manage_restaurants(self.test_file, self.test_file_backup)
        
        with open(self.test_file, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(len(data), 1) # Data should remain unchanged
        self.assertIn("Operation cancelled.", mock_stdout.getvalue())
        
if __name__ == "__main__":
    # unittest.main() # Unit Test
    manage_restaurants(FILEPATH, BACKUP_PATH) # Actual UI Call

