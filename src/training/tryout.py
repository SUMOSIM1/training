import training.parallel as pr
import training.sgym.qlearn as ql
import training.simrunner as sr
import pprint as pp


def main():
    pcfg = pr.ParallelConfig.Q_RW_0
    ptcs1 = pr.create_train_configs1(pcfg, 10)
    pp.pprint(ptcs1)

    for i, ptcs in enumerate(ptcs1):
        for ptc in ptcs:
            qlc = ql.parallel_to_qtrain_config(ptc)
            # print(f"{i} -- {pp.pformat(qlc)}")
            reward_handler = sr.RewardHandlerProvider.get(
                sr.RewardHandlerName(qlc.reward_handler_name)
            )
            print(f"{i} -- {ptc.name} {reward_handler}")
