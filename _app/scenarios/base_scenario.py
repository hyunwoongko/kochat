"""
@auther Hyunwoong
@since {6/21/2020}
@see : https://github.com/gusdnd852
"""


class BaseScenario:

    def _check_entity(self, text, entity, dict_):
        for t, e in zip(text, entity):
            for k, v in dict_.items():
                if k in e:
                    v.append((t, e))

        return dict_

    def _set_as_default(self, list_, default):
        if len(list_) == 0:
            list_.append(default)

        return list_
