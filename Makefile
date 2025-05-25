.PHONY: venv install clean prep train docker

# 1. Удалить всё
clean:
	rm -rf venv
	rm -rf models
	rm -rf logs
	rm -rf catboost_info
	rm -rf notebooks/catboost_info
	rm -rf notebooks/model
	rm -rf .DS_Store
	rm -f features/player_features.csv
	docker-compose down -v --remove-orphans

# 2. Создание виртуального окружения
venv:
	python3 -m venv venv

# 3. Установка зависимостей в venv
install: venv
	venv/bin/pip install --upgrade pip
	venv/bin/pip install -r requirements.txt

# 4. Подготовка данных (с использованием Python из venv)
prep:
	venv/bin/python src/prepdata.py

# 5. Обучение моделей
train:
	venv/bin/python src/fit.py

# 6. Работа с Docker
docker:
	docker-compose build --no-cache
	docker-compose up
