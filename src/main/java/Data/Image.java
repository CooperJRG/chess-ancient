package Data;

public class Image {

    private double[][] data;
    private int label;

    public Image(double[][] data, int label) {
        this.data = data;
        this.label = label;
    }

    public double[][] getData() {
        return data;
    }

    public int getLabel() {
        return label;
    }

    @Override
    public String toString(){
        StringBuilder sb = new StringBuilder();
        sb.append(label).append(", \n");

        for (double[] datum : data) {
            for (int j = 0; j < data[0].length; j++) {
                sb.append(datum[j]).append(", ");
            }
            sb.append("\n");
        }

        return sb.toString();
    }
}
