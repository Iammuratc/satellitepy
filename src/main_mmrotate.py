from settings.mmrotate import SettingsMMRotate
from data.eval_mmrotate import EvalMMRotate

exp_name = 'rotated_retinanet_0'

settings = SettingsMMRotate(		
	exp_name=exp_name,
	exp_root='mmrotate/work_dirs',
	test_pkl='test.pkl')()

my_ind = 1
evaluate = EvalMMRotate(settings)
# print(evaluate.df.iloc[my_ind,7])
# evaluate.show_image(my_ind)

