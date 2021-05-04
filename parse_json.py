import json
import os
import pandas

csv_file=open("tos_data.csv", "w")
csv_file.write("label,text\n")

for filename in os.listdir("json_files"):
    if filename.endswith(".json"):
        with open("json_files/"+filename, "r") as read_it:
            data=json.load(read_it)
            for k,v in data["pointsData"].items():
                if("quoteText" in v.keys() and "tosdr" in v.keys() and "point" in v["tosdr"].keys() and v["quoteText"]!=""):
                    if(v["tosdr"]["point"]=="good"):
                        csv_file.write("2," + v["quoteText"].replace(',','').replace('\n','') + "\n")
                    if(v["tosdr"]["point"]=="neutral"):
                        csv_file.write("1," + v["quoteText"].replace(',','').replace('\n','') + "\n")
                    if(v["tosdr"]["point"]=="bad"):
                        csv_file.write("0," + v["quoteText"].replace(',','').replace('\n','') + "\n")
csv_file.close()
