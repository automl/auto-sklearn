cd AutoML2015
python run.py ~/projects/automl_competition_2015/datasets/000/ ~/tmp/automl2015

# Uncomment to profile run.py
#cd AutoML2015
#python -m cProfile -o .profile.pstats run.py ~/projects/automl_competition_2015/datasets/000/ ~/tmp/automl2015
#python -c "import pstats; p = pstats.Stats('.profile.pstats'); p.sort_stats('cumulative').print_stats('AutoSklearn', 50)"
