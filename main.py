import crear_cromosoma as cc


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    chromosome = cc.generate(100, 15)
    for row in chromosome:
        print(row)
    print("[{}, {}]".format(len(chromosome), len(chromosome[0])))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

# 1. Medir distacias entre puntos (usar la formula)y graficar la mejor
