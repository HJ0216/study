import java.util.List;
import java.util.Scanner;

public class MemberInsert implements Member {

	Scanner scan = new Scanner(System.in);
	MemberService mService = new MemberService();
	
	@Override
	public void execute(List<MemberDTO_IO> list) {
		System.out.print("\nEnter the Name: ");
		String name = scan.next();

		System.out.print("Enter the age: ");
		int age = scan.nextInt();
		
		System.out.print("Enter the Phone number: ");
		String phone_num = scan.next();
		
		System.out.print("Enter the Address: ");
		String address = scan.next();

		MemberDTO_IO mDTO = new MemberDTO_IO(name, age, phone_num, address);
		
		list.add(mDTO);

		System.out.println("Saved the data");
		
	} // execute()
	
}
