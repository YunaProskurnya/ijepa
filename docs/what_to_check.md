# What to check

## Подсчет erank

Проверить предложение Ивана об использовании для подсчёта еранка сингулярных
чисел матрицы ковариации, а не центрированной исходной матрицы признаков.
Для этого надо удалить `.sqrt()` в двух местах.

В [src/representation_diagnostics.py](../src/representation_diagnostics.py) сейчас:

```python
    singular_values = eigenvalues[:min(n - 1, d)].clamp_min(0).sqrt()
```

Для проверки:

```python
    singular_values = eigenvalues[:min(n - 1, d)].clamp_min(0)  # .sqrt()
```

В [scripts/evaluate_knn.py](../scripts/evaluate_knn.py) сейчас:

```python
    singular_values = eigenvalues[eigenvalues > tolerance].sqrt()
```

Для проверки:

```python
    singular_values = eigenvalues[eigenvalues > tolerance]  # .sqrt()
```
