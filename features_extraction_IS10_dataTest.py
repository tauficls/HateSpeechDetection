import pandas as pd
import os

script_dir = os.path.dirname(__file__)
# temp = 0
def config(path, filename):
	
	kelas = filename.split("_")
	if kelas[0] == "H":
		classes = 1
	else:
		classes = 0
	
	os.system("SMILExtract_Release -C E:/TugasAkhir/HateSpeechModel/hate_config/IS10_paraling.conf -I " + path + filename
		+ " -lldarffoutput E:/TugasAkhir/HateSpeechModel/hatespeechweb/static/arff/IS10_paraling_lld_test.arff " 
		+ "-output E:/TugasAkhir/HateSpeechModel/hatespeechweb/static/arff/IS10_paraling_func_test.arff -class " 
		+ str(classes) + " -N " + filename.split(".")[0])
