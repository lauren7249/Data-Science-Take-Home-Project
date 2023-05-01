import neptune
import os

# Create a Neptune run object
run = neptune.init_run(
    project="lauren7249/Tesorio-Take-Home",
    api_token=os.getenv('NEPTUNE_API_TOKEN'))