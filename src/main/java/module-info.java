module com.example.chessv2 {
    requires javafx.controls;
    requires javafx.fxml;
    requires javafx.web;

    requires org.controlsfx.controls;
    requires com.dlsc.formsfx;
    requires org.kordamp.ikonli.javafx;
    requires org.kordamp.bootstrapfx.core;
    requires eu.hansolo.tilesfx;
    requires com.almasb.fxgl.all;
    requires java.sql;
    requires mapdb;
    requires deeplearning4j.nn;
    requires deeplearning4j.parallel.wrapper;
    requires nd4j.api;
    requires slf4j.api;
    requires deeplearning4j.core;
    requires deeplearning4j.vertx;
    requires deeplearning4j.ui.model;

    opens com.example.chessv2 to javafx.fxml;
    exports com.example.chessv2;
}