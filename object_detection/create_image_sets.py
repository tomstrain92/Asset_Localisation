import os, sys
from shutil import copy2
import argparse

sys.path.insert(0,'..')
from survey import Survey 
from progress.bar import Bar


def ensure_dir(directory):
    if not os.path.exists(directory):
            os.makedirs(directory)


def main(args):

	survey = Survey(args.data_dir, args.road)
	# Refmarkers are common and other assets are often near
	#MKRF = survey.load_assets('MKRF')
	#RLLP = survey.load_assets('RLLP')
	SNSF = survey.load_assets('SNSF')

	set_num = 0
	out_dir = os.path.join(args.data_dir, args.road, "Asset_Sets", "Set_{}".format(set_num))
	ensure_dir(out_dir)

	image_num = 0

	# fist progress
	bar = Bar('creating set {}'.format(set_num), max=args.set_size)

	for ind, image in survey.nav_file.iterrows():
		# is there an asset infront of the image?
		#_, MKRF_bool = survey.find_close_assets(MKRF, image)
		#_, RLLP_bool = survey.find_close_assets(RLLP, image)
		_, SNSF_bool = survey.find_close_assets(SNSF, image)
		
		# create new set if current is full
		if image_num > args.set_size:
			print('\n[INFO] {:.1f} % complete'.format(100 * ind/len(survey.nav_file)))
			set_num += 1
			out_dir = os.path.join(args.data_dir, args.road, "Asset_Sets", "Set_{}".format(set_num))
			ensure_dir(out_dir)
			image_num = 0
			bar.finish()
			bar = Bar('creating set {}'.format(set_num), max=args.set_size)

		if SNSF_bool:
			if(ind % args.step_size == 0):
				image_file = os.path.join(args.data_dir, args.road, "Images",
									"2_{}_{}.jpg".format(image["PCDATE"], image["PCTIME"]))
				copy2(image_file, out_dir)
				image_num += 1
				bar.next()




if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Asset location tool')
	parser.add_argument("road", type=str,
						help="road/highway to split")
	parser.add_argument("--data_dir", "-d", type=str, default="/media/tom/Elements", 
						help="top level data directory")
	parser.add_argument("--set_size", type=int, default=1000,
						help="number of images in each set")
	parser.add_argument("--step_size", type=int, default=1,
						help="how many images to skip per step, consequetive images will be similar")

	args = parser.parse_args()

	main(args)