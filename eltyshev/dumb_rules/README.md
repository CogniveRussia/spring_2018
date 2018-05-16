### Поиск ручных правил для False Positive

Было замечено, что некоторые False Positive алерты могут быть отсеяны с помощью ручных правил. В данном модуле находится алгортим поиска данных ручных правил

Внутри каждого критерия рассматриваются следующие показатели: ["P_BRANCH", "P_CURRENCYCODE", "P_EKNPCODE", "P_DOCCATEGORY", "P_SUSPIC_KIND", "P_CRITERIAFIRST", "P_CRITERIASECOND"] и ищутся правила вида column=some_value -> False Positive

или P_EKNPCODE == 390, которые однозначно идентифицируют алерт как False Positive и покрывают 150142 из 326505 (48%) FP случаев
