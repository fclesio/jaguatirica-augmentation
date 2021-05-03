destroy:
	echo "Destroy all non-source folders"
	rm -rf src/main/data/reshaped/
	rm -rf src/main/data/train_augmented/
	rm -rf src/main/data/test_augmented/
	rm -rf src/main/data/validation_augmented/
	echo "Create all non-source folders"
	mkdir src/main/data/reshaped
	mkdir src/main/data/train_augmented
	mkdir src/main/data/test_augmented
	mkdir src/main/data/validation_augmented

