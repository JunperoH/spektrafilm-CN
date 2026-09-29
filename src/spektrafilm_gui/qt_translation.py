"""Load Qt's bundled Chinese catalogs for standard dialogs and buttons."""

from __future__ import annotations

from qtpy import QtCore, QtWidgets


_TRANSLATORS: list[QtCore.QTranslator] = []


def install_qt_chinese() -> None:
    QtCore.QLocale.setDefault(QtCore.QLocale('zh_CN'))
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    catalog_dir = QtCore.QLibraryInfo.path(QtCore.QLibraryInfo.TranslationsPath)
    for catalog in ('qt_zh_CN', 'qtbase_zh_CN'):
        translator = QtCore.QTranslator(app)
        if translator.load(catalog, catalog_dir):
            app.installTranslator(translator)
            _TRANSLATORS.append(translator)
