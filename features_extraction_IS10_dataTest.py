import pandas as pd
import os

script_dir = os.path.dirname(__file__)
# temp = 0
def config(filename):
	

	if (data["Kelas"] == 0):
		kelas = "NH"
	else:
		kelas = "H"

	if (data["No"] < 10):
		classes = kelas + "_00" + str(data["No"])
		os.system("SMILExtract_Release -C E:/TugasAkhir/HateSpeechModel/hate_config/IS10_paraling.conf -I " + filename + classes 
			+ ".wav" + " -lldarffoutput E:/TugasAkhir/HateSpeechModel/arff/Interspeech/IS10_paraling_lld_test.arff " 
			+ "-output E:/TugasAkhir/HateSpeechModel/arff/Interspeech/IS10_paraling_func_test.arff -class " + str(data["Kelas"]) + " -N " + classes)
	elif (data["No"] < 100):
		classes = kelas + "_0" + str(data["No"])
		os.system("SMILExtract_Release -C E:/TugasAkhir/HateSpeechModel/hate_config/IS10_paraling.conf -I E:/TugasAkhir/DataSpeechTesting/" + classes 
			+ ".wav" + " -lldarffoutput E:/TugasAkhir/HateSpeechModel/arff/Interspeech/IS10_paraling_lld_test.arff "
			+ "-output E:/TugasAkhir/HateSpeechModel/arff/Interspeech/IS10_paraling_func_test.arff -class " + str(data["Kelas"]) + " -N " + classes)
	else:
		classes = kelas + "_" + str(data["No"])
		os.system("SMILExtract_Release -C E:/TugasAkhir/HateSpeechModel/hate_config/IS10_paraling.conf -I E:/TugasAkhir/DataSpeechTesting/" + classes
			+ ".wav" + " -lldarffoutput E:/TugasAkhir/HateSpeechModel/arff/Interspeech/IS10_paraling_lld_test.arff " 
			+ "-output E:/TugasAkhir/HateSpeechModel/arff/Interspeech/IS10_paraling_func_test.arff -class " + str(data["Kelas"]) + " -N " + classes)
