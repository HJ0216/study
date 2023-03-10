import java.util.*;

abstract class Shape {
	protected double area;
	protected Scanner scan = new Scanner(System.in);
	
	public Shape() {System.out.println("Shape Default Constructor");}
	public abstract void calcArea(); // 쓰이지 않는 method는 abstract화
	public abstract void dispArea();
}


class Triangle extends Shape { // area, calcArea(), dipArea() 사용 가능
	protected int base, height;
	
	public Triangle() {
		System.out.println("Triangle Default Constructor");
		System.out.print("Base: ");
		base = scan.nextInt();
		System.out.print("Height: ");
		height = scan.nextInt();
	}
	
	// @ Annotation
	@Override
	public void calcArea() {
		area = base * height / 2.0;
	}
	
	@Override
	public void dispArea() {
		System.out.println("Triangle area: " + area);
	}
}

class Square extends Shape {
	protected int width, height;
	
	public Square() {
		System.out.println("Square Default Constructor");
		System.out.print("Width: ");
		width = scan.nextInt();
		System.out.print("Height: ");
		height = scan.nextInt();		
	}
	
	// @ Annotation
	@Override
	public void calcArea() {
		area = width * height;
	}

	@Override
	public void dispArea() {
		System.out.println("Square area: " + area);
	}

}

class Trapezoid extends Shape {
	protected int top, bottom, height;
	
	public Trapezoid() {
		System.out.println("Trapezoid Default Constructor");
		System.out.print("Top: ");
		top = scan.nextInt();
		System.out.print("Bottom: ");
		bottom = scan.nextInt();
		System.out.print("Height: ");
		height = scan.nextInt();		
	}

	@Override
	public void calcArea() {
		area = (top+bottom) * height / 2.0;
	}

	@Override
	public void dispArea() {
		System.out.println("Trapezoid area: " + area);
	}
	
}


public class ShapeMain_abs {

	public static void main(String[] args) {
		// 다형성: 부모 참조변수는 자식 객체를 참조할 수 있음
		Shape shape; // Parent remote controller create
		shape = new Triangle();
		shape.calcArea();
		shape.dispArea();
		System.out.println();

		
		shape = new Square();
		shape.calcArea();
		shape.dispArea();
		System.out.println();
		
		
		shape = new Trapezoid();
		shape.calcArea();
		shape.dispArea();
		System.out.println();
	}
}
