package com.gatech.yan.java_weka_jmathplot;

import java.util.Objects;

public class Point {
    public double x;
    public double y;

    public Point(double first, double second) {
        this.x = first;
        this.y = second;
    }
    
    

	public double getX() {
		return x;
	}



	public void setX(double x) {
		this.x = x;
	}



	public double getY() {
		return y;
	}



	public void setY(double y) {
		this.y = y;
	}



	@Override
	public String toString() {
		return "Point ["+x + "," + y +"]";
	}

	@Override
	public int hashCode() {
		return Objects.hash(x, y);
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Point other = (Point) obj;
		return Double.doubleToLongBits(x) == Double.doubleToLongBits(other.x)
				&& Double.doubleToLongBits(y) == Double.doubleToLongBits(other.y);
	}
    
    
}