from setuptools import setup

package_name = 'gsl_bench'

setup(
    name=package_name,
    version='0.1.0',
    packages=[
        package_name,
        f'{package_name}.harness',
        f'{package_name}.eval',
        f'{package_name}.agents',
    ],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/gsl_bench']),
        ('share/gsl_bench', ['package.xml']),
        ('share/gsl_bench/configs', ['configs/harness_default.yaml']),
        ('share/gsl_bench/configs/agents', [
            'configs/agents/gpulaika.yaml', 'configs/agents/adsm.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='efe',
    maintainer_email='efemantaroglu@gmail.com',
    description='Modular GSL benchmark: implement observe()/act(), the harness does the rest.',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'runner_node = gsl_bench.harness.runner_node:main',
            'eval = gsl_bench.eval.episode_runner:main',
            'report = gsl_bench.eval.report:main',
            'oracle = gsl_bench.eval.oracle:main',
            'world_align = gsl_bench.eval.world_align:main',
        ],
        # Third parties register their agents under this group from their own package.
        'gsl_bench.agents': [
            'random_walk = gsl_bench.agents.random_walk:RandomWalkAgent',
            'upwind_greedy = gsl_bench.agents.upwind_greedy:UpwindGreedyAgent',
            'gpulaika = gsl_bench.agents.gpulaika_agent:GpulaikaAgent',
            'zigzag = gsl_bench.agents.zigzag:ZigzagAgent',
            'surge_cast = gsl_bench.agents.surge_cast:SurgeCastAgent',
            'adsm = gsl_bench.agents.adsm_agent:AdsmAgent',
        ],
    },
)
