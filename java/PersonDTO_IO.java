import java.io.Serializable;

public class PersonDTO_IO implements Serializable { // abstract method X

	private String name;
	private int age;
	private double height;

	public PersonDTO_IO(String name, int age, double height) {
		this.name = name;
		this.age = age;
		this.height = height;
	}
	
	public void setName(String name) {this.name = name;}
	public void setAge(int age) {this.age = age;}
	public void setHeight(double height) {this.height = height;}
	
	public String getName() {return name;}
	public int getAge() {return age;}
	public double getHeight() {return height;}
	
}
