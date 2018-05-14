Задача: построить TDA
Данные: сэмпл (82196 транзакций), сформированный из таблицы offline_operations. 
Признаковое описание:
"Сырые" признаки (взяты из offline_operations): P_BASEAMOUNT, P_EKNPCODE, P_CURRENCYCODE, P_DOCCATEGORY, P_KFM_OPER_REASON
Сгенерированные статистики: движение средств по дебитному и кредитному счетам по разрезам

Методы: в качестве линзы использовался PCA (n_dim=2 и 3), а также выходы IsolationForest'a.
Для кластеризации использовались KMeans и AffinityPropagation. В итоге было построено 4 модели:
- Isolation Forest + KMeans
- Isolation Forest + AffinityPropagation
- PCA + KMeans
- PCA + AffinityPropagation

Итоговые визуализации созданы с помощью kepler mapper и сохранены в папку graphs.

P.S. Поскольку AffinityPropagation обучается долго, для моделей 2 и 4 никакого подбора параметров не производилось.
