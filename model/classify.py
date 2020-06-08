import tensorflow as tf
import base64
from datetime import datetime
import pytz

# Object detection imports
from model.utils import backbone
from model.api import object_counting_api

# Custom imports
import model.odata_call as odata
import model.get_config as config

# Initializing Variables for inference
fps = 30 # change it with your input fps
width = 640 # change it with your input width
height = 480 # change it with your input height
is_color_recognition_enabled = 0

# Initializing Variables
WorkOrder = "RED"
NestId = "NST001"


Underload = config.get_load("Underload")
Overload = config.get_load("Overload")
Nest_Capacity = config.get_load("Nest_Capacity")

def predict(image_string):
    # Get the Load No
    LoadNo = odata.get_load_no()

    # Read the image_data
    imgdata = base64.b64decode(image_string)
    with open('./model/images/image.jpg', 'wb') as f:
        f.write(imgdata)

    # Input Image
    image_data = "./model/images/image.jpg"

    detection_graph, category_index = backbone.set_model('./model/inference_graph')
    print(detection_graph,category_index)

    # TensorFlow Inference
    result = object_counting_api.single_image_object_counting(image_data, detection_graph, category_index, is_color_recognition_enabled, fps, width, height)
    #print (result)

    # Getting the Material Number from material config file 
    mat = odata.get_material_no(result)
    print(mat)

    # Getting the Material Data from Material Master
    mat_data = odata.material_data(' or '.join("{!s}".format(key) for (key) in mat.keys()))

    # Adding Quantity to the dataframe
    mat_data["Qty"] = mat_data["MaterialNo"].str.lstrip("0").astype(int).map(mat)
    mat_data["Area"] = (mat_data["Area"].astype(float) * mat_data["Qty"]).round(3)
    mat_data["Area"] = mat_data["Area"].astype(str)
    mat_data['Qty'] = mat_data['Qty'].astype(str)
    mat_data.insert(0, 'LoadNo', LoadNo)
    mat_data.insert(1, 'NestId', NestId)
    currDate = datetime.now(pytz.utc).astimezone(pytz.timezone('US/Eastern'))
    mat_data.insert(2, 'Cdate', currDate.strftime('%Y-%m-%dT%H:%M:%S'))

    print("\nNest Details:\n", mat_data)

    Total_Area = mat_data["Area"].astype(float).sum().round(2)
    T_Area = (Total_Area/Nest_Capacity * 100).round(2)
    print("\nCurrent Nest Loading Level: ", T_Area,"%")

    # Datafram for correct and wrong parts
    correct, wrong = odata.check_data(mat_data, WorkOrder)

    # Posting in S4 if correct / Showing part in case of wrong
    if wrong.empty:
        image = "./model/output_images/image.jpg"

        posting(correct, Total_Area, currDate, image, LoadNo)

    else:
        # To show the wrong parts    
        temp = "{" + result + "}"
        temp = eval(temp)
        list_target = []
        for k, v in temp.items():
            if (wrong["MaterialNo"].str.lstrip("0").astype(int) == config.get_material(k)).any():
                list_target.append(k)
        targeted_objects = ', '.join(list_target)

        print("\nRemove these objects:\n")
        result, image = object_counting_api.single_image_target_counting(image_data, detection_graph, category_index, is_color_recognition_enabled, targeted_objects, fps, width, height) # targeted objects counting
        
        image = "./model/output_images/wrong_image.jpg"

        posting(mat_data, Total_Area, currDate, image, LoadNo)

    return(result)

def posting(mat_tab, Total_Area, currDate, image, LoadNo):
        
        final = mat_tab.to_json(orient='records')
        final = final.replace("[",'{"MaterialNo": "000000","Area": "","Color": "","ZDetailToItem": [').replace("]","]}")
        print(final)
        
        #Converting Image from JPG to base64 and sending to S4 in Binary format
        image = odata.convert_image(image)
        
        # For posting image in S4
        odata.post_image("", "", image, "")
        
        # For posting Nest data in S4
        odata.post_data(final)
        
        # For posting Consumption data in S4
        Energy_Target = config.get_consump("Energy_Target")
        Energy_Meter = config.get_consump("Energy_Meter")
        Propene_Target = config.get_consump("Propene_Target")
        Propene_Meter = config.get_consump("Propene_Meter")
        
        # Target Consumption Rate
        Targetted_Energy = Total_Area * Energy_Target
        Targetted_Propene = Total_Area * Propene_Target
        
        # Actual Consumption Rate
        elapsedTime = currDate - datetime(2019, 4, 1, 10, 00, 00).astimezone(pytz.timezone('US/Eastern'))
        minutes, sec = divmod(elapsedTime.days * 86400 + elapsedTime.seconds, 60)
        hours = minutes / 60
        
        Actual_Energy = (hours * Energy_Meter)
        Actual_Energy = round(Actual_Energy, 2)
        Actual_Propene = (hours * Propene_Meter)
        Actual_Propene = round(Actual_Propene, 2)
        
        Date = str(currDate.strftime('%Y-%m-%dT00:00:00'))
        Time = str(currDate.strftime('PT%HH%MM%SS'))
        Total_Area = str(Total_Area)
        Targetted_Energy = str(Targetted_Energy)
        Targetted_Propene = str(Targetted_Propene)
        Actual_Energy = str(Actual_Energy)
        Actual_Propene = str(Actual_Propene)
        
        odata.post_consump(Date, Time, LoadNo, "ENERGY", Total_Area, Targetted_Energy, Actual_Energy)
        odata.post_consump(Date, Time, LoadNo, "PROPENE", Total_Area, Targetted_Propene, Actual_Propene)
        